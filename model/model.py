from transformers import PretrainedConfig
import torch
from torch import nn 
from typing import Optional
import math

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
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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

        self.num_key_value_heads = args.num_key_value_heads
        