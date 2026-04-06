from transformers import PretrainedConfig
import torch
from torch import nn 
from typing import Optional, Tuple
import math
from torch.nn import functional as F
from .activation_functions import ACT2FN

# huggingface config
class RaysMindConfig(PretrainedConfig):
    model_type = "raysmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


# RMSNorm
class RMSNorm(nn.Module):
    # init
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # norm
    def _norm(self,x:torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # forward
    def forward(self, x: torch.Tensor):
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int(32*1024), rope_base, rope_scaling: Optional[dict]=None):
    #init rope frequencies 
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2]).float()/dim)
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = rope_scaling['original_max_position_embeddings'], rope_scaling['factor'], rope_scaling['beta_fast'], rope_scaling['beta_slow']

    # if inference length is larger than original max, scale the frequency
    if end > orig_max:
        # mapping from ratio b to dimension index
        inv_dim = lambda b:  (dim*math.log(orig_max/(b*2*math.pi))) / (2* math.log(rope_base))

            # calculate the split point of high frequency and low frequency
        low, high = (
            max(math.floor(inv_dim(beta_fast)), 0),
            min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
        )
        # calculate the ramp factor
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=freqs.device).float() - low)
            / max(high - low, 0.001),
            0,
            1,
        )
        # frequency fusion formula: f'(i) = f(i) * ((1-γ) + γ/s)
        freqs = freqs * (1 - ramp + ramp / factor)

    # generate the position index vector t
    t = torch.arange(end, device=freqs.device)

    # calculate the outer product
    freqs = torch.outer(t, freqs).float()

    # calculate the Cos and Sin, and apply the attention compensation factor (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin

# RoPE
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # [a,b] -> [-b, a]
    def rotate_half(x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat([-x2,x1], dim=-1)
    
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (x[:,:,:,None,:].expand(bs, slen, num_key_value_heads, n_rep, head_dim)).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)

class Attention(nn.Moudle):
    def __init__(self, args: RaysMindConfig):
        super().__init__()

        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads
        assert self.num_key_value_heads % args.num_attention_heads == 0, "num_key_value_heads must be divisible by num_attention_heads"

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention)

    
    def forward(self, x: torch.Tensor, 
        position_embedding: Tuple[torch.Tensor, torch.Tensor], 
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        use_cache: bool = False, 
        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # compute Q, K, V
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # split Q, K, V into local heads by view
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # use rope for Q and K
        cos, sin = position_embedding
        xq, xk = apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])
        # use repeat_kv for K, V (note: kv cache)
        
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1,2),
            repeat_kv(xk, self.n_rep).transpose(1,2),
            repeat_kv(xv, self.n_rep).transpose(1,2)
            )
        
        # compute attention scores, Q@K^T/sqrt(d)
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (None if attention_mask is None else attention_mask.view(bsz,1,1,-1).expand(bsz, self.n_local_heads, seq_len, -1).bool())
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, 
            dropout_p=self.dropout if self.training else 0, is_causal=True)

        else:
            scores = (xq @ xk.transpose(-2, -1)) * (1 / math.sqrt(self.head_dim))
            scores = scores + torch.triu(torch.full((seq_len, seq_len), float('-inf'),device=scores.device), diagnonal=1).unsqueeze(0).unsqueeze(0)
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
        # concat and apply softmax
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores) 
        output = scores @ xv
        # [gsz, n_local_heads, seq_len, head_dim] -> [gsz, seq_len, n_local_heads * head_dim]
        output = output.transpose(1,2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
        
        
class FeedForward(nn.Module):
    # 初始化
    # 升维
    # 降维
    # 门控
    # dropout
    # 激活函数

    def __init__(self, args: RaysMindConfig):
        super().__init__()

        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size*8/3)
            args.intermediate_size = 64*((intermediate_size+(64-1))//64)

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False) 
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)))










        