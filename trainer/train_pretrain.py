import os
import sys


__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse  # CLI argument parsing
import time  # Timing
import warnings  # Warning filters
import torch
import torch.distributed as dist  # Distributed training
from contextlib import nullcontext  # Context manager
from torch import optim  # Optimizer
from torch.nn.parallel import DistributedDataParallel  # DDP
from torch.utils.data import DataLoader, DistributedSampler  # Data loading

from model.model import RaysMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (  # Training utilities
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()  # Wall-clock start

    # Iterate batches
    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(
            args.device
        )  # Move attention_mask to device

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            # Forward
            res = model(
                input_ids, labels=labels, attention_mask=attention_mask
            )  # Loss computed inside the model

            loss = (
                res.loss + res.aux_loss
            )  # Model loss (replaces manual loss_fct + mask)

            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            # Unscale gradients to true magnitudes before clipping
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # GradScaler: step() updates weights; update() adjusts scale
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # Undo accum scaling
            current_lr = optimizer.param_groups[-1]["lr"]

            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )

            # Experiment tracking
            if wandb:
                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()

            moe_suffix = (
                "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
            )
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

            # DDP: read state from .module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # fp16 checkpoint to save disk
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)

            # Full resume blob (optimizer, scaler, step, …)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",  # Relative to cwd
            )

            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RaysMind Pretraining")

    # ----- Core training -----
    parser.add_argument(
        "--save_dir", type=str, default="../out", help="Directory for saved weights"
    )
    parser.add_argument(
        "--save_weight", default="pretrain", type=str, help="Checkpoint filename prefix"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs (e.g. 1 smoke test, 2–6 full run)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial LR")

    # ----- Hardware & perf -----
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="AMP dtype (bfloat16 or float16)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="DataLoader worker processes"
    )

    # ----- Training strategy -----
    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip norm")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument(
        "--save_interval", type=int, default=100, help="Save weights every N steps"
    )

    # ----- Model architecture -----
    parser.add_argument("--hidden_size", default=512, type=int, help="Hidden size")
    parser.add_argument(
        "--num_hidden_layers", default=8, type=int, help="Number of transformer layers"
    )
    parser.add_argument(
        "--max_seq_len", default=512, type=int, help="Max sequence length (truncate)"
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="Use MoE MLP (0=no, 1=yes)",
    )

    # ----- Data & resume -----
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/pretrain_hq.jsonl",
        help="Pretraining JSONL path",
    )
    parser.add_argument(
        "--from_weight",
        default="none",
        type=str,
        help="Init from checkpoint tag (none = from scratch)",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="Auto load resume checkpoint (0=no, 1=yes)",
    )

    # ----- Experiment tracking -----
    parser.add_argument("--use_wandb", action="store_true", help="Enable SwanLab / W&B")
    parser.add_argument(
        "--wandb_project", type=str, default="RaysMind-Pretrain", help="Project name"
    )

    args = parser.parse_args()

    # ----- 1. Distributed & seeds -----
    """
    Distributed:
    - local_rank: GPU index on this machine
    - Per-rank seed: reproducible but not identical shuffles across ranks
    """
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    # Base seed 42 + rank
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ----- 2. Dirs, config, checkpoint -----
    """
    Config & checkpoints:
    - Ensure save_dir exists
    - Build lm_config
    - Optionally load resume dict
    """
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = RaysMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ----- 3. Mixed precision -----
    """
    AMP:
    - bfloat16: wide dynamic range, often stable
    - float16: less memory; GradScaler helps
    - autocast: lower precision for eligible ops
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # No autocast on CPU
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ----- 4. W&B / SwanLab -----
    """
    Tracking: SwanLab imported as wandb; resume via wandb_id in checkpoint
    """
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None

        wandb_run_name = f"RaysMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ----- 5. Model, data, optimizer -----
    """
    Stack: init_model, PretrainDataset, DistributedSampler, AdamW, GradScaler (fp16)
    """
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        # RoPE tables: no grads, skip DDP broadcast
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: skipping first {start_step} steps, starting at step {start_step + 1}"
            )
            train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)
