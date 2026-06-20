from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.dataset import build_loaders
from src.metrics import compute_miou
from src.model import build_model
from src.utils import load_checkpoint, save_checkpoint, visualize_predictions


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best = None

    def step(self, metric: float) -> bool:
        if self.best is None or metric > self.best:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_criterion(cfg: dict):
    ignore_index = cfg["training"]["ignore_index"]
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
    w_ce = cfg["training"]["ce_weight"]
    w_dice = cfg["training"]["dice_weight"]

    def criterion(logits, targets):
        return w_ce * ce(logits, targets) + w_dice * dice(logits, targets)

    return criterion


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, cfg, device):
    model.train()
    total_loss = 0.0
    log_every = cfg["logging"]["log_every_n_steps"]
    use_amp = cfg["training"]["mixed_precision"]

    pbar = tqdm(loader, desc="  train", leave=False)
    for step, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        if (step + 1) % log_every == 0:
            avg = total_loss / (step + 1)
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, cfg, device):
    model.eval()
    total_loss = 0.0
    all_ious = []
    use_amp = cfg["training"]["mixed_precision"]

    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        miou, _ = compute_miou(preds, masks, num_classes=cfg["model"]["num_classes"])
        all_ious.append(miou)

    mean_loss = total_loss / len(loader)
    mean_miou = torch.stack(all_ious).mean().item()
    return mean_loss, mean_miou


def train(cfg: dict, resume_from: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)

    t_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"]
    )
    total_steps = t_cfg["epochs"] * len(train_loader)
    warmup_steps = t_cfg["warmup_epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=t_cfg["mixed_precision"])
    early_stop = EarlyStopping(patience=t_cfg["early_stop_patience"])

    start_epoch = 0
    best_miou = 0.0
    ckpt_dir = Path(cfg["checkpointing"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_from:
        print(f"Resuming from {resume_from}")
        start_epoch, best_miou = load_checkpoint(resume_from, model, optimizer, device)
        start_epoch += 1

    for epoch in range(start_epoch, t_cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{t_cfg['epochs']} — best mIoU so far: {best_miou:.4f}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, cfg, device
        )
        val_loss, val_miou = validate(model, val_loader, criterion, cfg, device)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}")

        # always save last checkpoint to Drive
        last_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_miou": best_miou,
            "val_miou": val_miou,
        }
        local_last = Path("/tmp/last.pth")
        save_checkpoint(last_state, local_last, drive_path=ckpt_dir / "last.pth")

        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(
                last_state,
                local_last,
                drive_path=ckpt_dir / f"best_epoch{epoch + 1:03d}_miou{val_miou:.4f}.pth",
            )
            print(f"  *** New best mIoU: {best_miou:.4f} — checkpoint saved ***")

        if early_stop.step(val_miou):
            print(f"  Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")
    return model
