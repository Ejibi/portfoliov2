"""
train.py  —  AxisDeepLabV3+  (Multi‑GPU DDP + Single‑GPU Seamless Fallback)
"""

import argparse
import copy
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config  import Config
from dataset import AxisCarDamageDataset, train_transforms, val_transforms
from model   import AxisDeepLabV3Plus
from loss    import AxisSegLoss, DeepSupervisionLoss
from metrics import MetricAccumulator, format_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def setup_distributed():
    """Initialise process group for DDP if launched via torchrun, else fall back to single-GPU."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    else:
        # Prevent stray environment variables from triggering distributed handshakes
        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)
        return 0, 1


def is_main_process(local_rank: int) -> bool:
    return local_rank == 0


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        raw_model = model.module if hasattr(model, "module") else model
        for s_param, m_param in zip(self.shadow.parameters(), raw_model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
            
        for (s_name, s_buf), (m_name, m_buf) in zip(self.shadow.named_buffers(), raw_model.named_buffers()):
            if "running_mean" in s_name or "running_var" in s_name or "num_batches_tracked" in s_name:
                s_buf.data.copy_(m_buf.data)
            elif s_buf.dtype in (torch.float16, torch.float32, torch.float64):
                s_buf.data.mul_(self.decay).add_(m_buf.data, alpha=1.0 - self.decay)
            else:
                s_buf.data.copy_(m_buf.data)

    def apply(self, model: nn.Module) -> None:
        raw_model = model.module if hasattr(model, "module") else model
        self._backup = copy.deepcopy(raw_model.state_dict())
        raw_model.load_state_dict(self.shadow.state_dict())

    def restore(self, model: nn.Module) -> None:
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(self._backup)


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(
    optimizer, warmup_epochs: int, total_epochs: int, min_lrs: list = [], base_lrs: list = []
) -> LambdaLR:
    lambdas = []
    for min_lr, base_lr in zip(min_lrs, base_lrs):
        min_ratio = min_lr / base_lr
        def lr_lambda(epoch: int, ratio=min_ratio) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * t))
        lambdas.append(lr_lambda)
    return LambdaLR(optimizer, lambdas)


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(
    model, loader, criterion, acc, optimizer, scaler, ema,
    cfg, device, training, world_size
):
    model.train(training)
    acc.reset()
    total_loss = total_dice = total_bce = total_edge = 0.0
    n = len(loader)

    if training and optimizer:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        use_amp = (scaler is not None) or (not training and cfg.use_amp)
        with autocast("cuda", enabled=use_amp):
            output = model(images)
            loss, info = criterion(output, masks)

        if training:
            loss = loss / cfg.grad_accum_steps
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0 or (batch_idx + 1 == n):
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)

        main_logits = output[0] if isinstance(output, tuple) else output
        acc.update(main_logits.detach(), masks)

        loss_val = (loss.item() * cfg.grad_accum_steps) if training else loss.item()
        total_loss += loss_val
        total_dice += info.get("dice", 0)
        total_bce += info.get("bce", 0)
        total_edge += info.get("edge", 0)

    if world_size > 1:
        for t in [acc._global_inter, acc._global_union, acc._global_total,
                  acc._global_tp, acc._global_fp, acc._global_fn]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "bce": total_bce / n,
        "edge": total_edge / n,
        **acc.compute(),
    }

# ── Main training function ────────────────────────────────────────────────────

def train(cfg: Config, local_rank: int, world_size: int) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if is_main_process(local_rank):
        log.info(f"Device: {device} | World size: {world_size} | Project: {cfg.project_name}")
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_ds = AxisCarDamageDataset(
        cfg.train_json, cfg.train_img_dir,
        transform=train_transforms(cfg.image_size),
        class_names=cfg.class_names, mask_dir=Path(cfg.train_mask_dir)
    )
    val_ds = AxisCarDamageDataset(
        cfg.val_json, cfg.val_img_dir,
        transform=val_transforms(cfg.image_size),
        class_names=cfg.class_names, mask_dir=Path(cfg.val_mask_dir)
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, sampler=val_sampler, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=True
    )

    if is_main_process(local_rank):
        log.info(f"Train {len(train_ds)} images  |  Val {len(val_ds)} images")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AxisDeepLabV3Plus(
        encoder_name=cfg.encoder_name, encoder_weights=cfg.encoder_weights,
        num_classes=cfg.num_classes, use_aux_head=cfg.use_aux_head, dropout=cfg.decoder_dropout,
    ).to(device)
    
    raw_model = model

    if world_size > 1:
        # find_unused_parameters=True is required to handle unused pretrained backbone layers
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )

    # ── Loss & Optimiser ──────────────────────────────────────────────────────
    base_loss = AxisSegLoss(
        dice_weight=cfg.dice_weight, bce_weight=cfg.bce_weight, edge_weight=cfg.edge_weight,
        ohem_keep_ratio=cfg.ohem_keep_ratio, label_smoothing=cfg.label_smoothing, smooth=cfg.smooth,
        focal_tversky_alpha=cfg.focal_tversky_alpha, focal_tversky_beta=cfg.focal_tversky_beta,
        focal_tversky_gamma=cfg.focal_tversky_gamma, pos_weight=cfg.pos_weight,
    )
    criterion = DeepSupervisionLoss(base_loss, aux_weight=cfg.aux_weight).to(device)

    optimizer = AdamW(raw_model.parameter_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))
    scheduler = cosine_schedule_with_warmup(
        optimizer, warmup_epochs=cfg.warmup_epochs, total_epochs=cfg.epochs,
        min_lrs=[cfg.min_lr, cfg.min_lr * 10], base_lrs=[cfg.lr_backbone, cfg.lr_head],
    )

    # ── AMP / EMA ─────────────────────────────────────────────────────────────
    scaler = GradScaler("cuda") if cfg.use_amp and device.type == "cuda" else None
    ema    = EMA(raw_model, cfg.ema_decay) if cfg.use_ema else None

    train_acc = MetricAccumulator(cfg.class_names, cfg.pred_threshold)
    val_acc   = MetricAccumulator(cfg.class_names, cfg.pred_threshold)

    best_miou = 0.0
    log_path  = cfg.log_dir / "train_log.csv"

    if is_main_process(local_rank):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_mean_iou,val_mean_dice,val_mean_bf1\n")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        if train_sampler: train_sampler.set_epoch(epoch)

        train_m = run_epoch(
            model, train_loader, criterion, train_acc,
            optimizer, scaler, ema, cfg, device, training=True, world_size=world_size
        )
        scheduler.step()

        # Switch to shadow weights for validation evaluation
        if ema: ema.apply(model)
        torch.cuda.empty_cache()
        
        val_m = run_epoch(
            model, val_loader, criterion, val_acc,
            None, None, None, cfg, device, training=False, world_size=world_size
        )
        
        if ema: ema.restore(model)

        # Logging and Checkpointing (Main Process Only)
        if is_main_process(local_rank):
            elapsed = time.time() - t0
            miou    = val_m.get("mean/iou", 0)
            mdice   = val_m.get("mean/dice", 0)
            mbf1    = val_m.get("mean/boundary_f1", 0)

            log.info(
                f"Epoch {epoch:3d}/{cfg.epochs}  "
                f"train_loss {train_m['loss']:.4f}  "
                f"val_loss {val_m['loss']:.4f}  "
                f"mIoU {miou:.4f}  mDice {mdice:.4f}  mBF1 {mbf1:.4f}  "
                f"[{elapsed:.1f}s]"
            )
            log.info(format_metrics(val_m, cfg.class_names))

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_m['loss']:.5f},{val_m['loss']:.5f},")
                f.write(f"{miou:.5f},{mdice:.5f},{mbf1:.5f}\n")

            if miou > best_miou:
                best_miou = miou
                save_model = ema.shadow if ema else raw_model
                torch.save({
                    "epoch":       epoch,
                    "model_state": save_model.state_dict(),
                    "val_metrics": val_m,
                    "cfg":         cfg,
                }, cfg.checkpoint_dir / "best.pt")
                log.info(f"  ✓ Saved best checkpoint  (mIoU {best_miou:.4f})")

        if world_size > 1:
            dist.barrier()

    if is_main_process(local_rank):
        log.info(f"Training complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    local_rank, world_size = setup_distributed()
    train(Config(), local_rank, world_size)