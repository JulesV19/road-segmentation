import copy
import csv
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.dataset import build_loaders
from src.evaluation import evaluate
from src.metrics import CLASS_NAMES
from src.model import build_model
from src.utils import load_checkpoint, save_checkpoint, wait_for_checkpoint_copies


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

    def state_dict(self) -> dict:
        return {"counter": self.counter, "best": self.best}

    def load_state_dict(self, state: dict):
        self.counter = state["counter"]
        self.best = state["best"]


def build_criterion(cfg: dict, device: str):
    ignore_index = cfg["training"]["ignore_index"]

    class_weights = cfg["training"].get("class_weights")
    weight_tensor = (
        torch.tensor(class_weights, dtype=torch.float32).to(device)
        if class_weights else None
    )

    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight_tensor)
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
        with torch.amp.autocast("cuda", enabled=use_amp):
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


def append_history(path: Path, row: dict):
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def train(cfg: dict, resume_from: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg, device)

    t_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"]
    )
    total_steps = t_cfg["epochs"] * len(train_loader)
    warmup_steps = t_cfg["warmup_epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=t_cfg["mixed_precision"])
    early_stop = EarlyStopping(patience=t_cfg["early_stop_patience"])

    start_epoch = 0
    best_miou = 0.0
    ckpt_dir = Path(cfg["checkpointing"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / "history.csv"

    if resume_from:
        print(f"Resuming from {resume_from}")
        ckpt = load_checkpoint(resume_from, model, optimizer, scheduler, scaler, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("best_miou", 0.0)
        if "early_stop" in ckpt:
            early_stop.load_state_dict(ckpt["early_stop"])

    for epoch in range(start_epoch, t_cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{t_cfg['epochs']} — best mIoU so far: {best_miou:.4f}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, cfg, device
        )
        val = evaluate(model, val_loader, cfg, device, criterion=criterion)
        val_miou = val["miou"]

        print(f"  train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  val_mIoU={val_miou:.4f}")

        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou
        stop = early_stop.step(val_miou)

        append_history(history_path, {
            "epoch": epoch + 1,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_miou": val_miou,
            **{f"iou_{name}": val["per_class_iou"][name] for name in CLASS_NAMES[1:]},
        })

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "early_stop": early_stop.state_dict(),
            "best_miou": best_miou,
            "val_miou": val_miou,
            "val_per_class_iou": val["per_class_iou"],
            "cfg": copy.deepcopy(cfg),
        }
        drive_paths = [ckpt_dir / "last.pth"] + ([ckpt_dir / "best.pth"] if is_best else [])
        save_checkpoint(state, Path("/tmp/last.pth"), drive_paths=drive_paths)
        if is_best:
            print(f"  *** New best mIoU: {best_miou:.4f} — checkpoint saved ***")

        if stop:
            print(f"  Early stopping triggered after {epoch + 1} epochs.")
            break

    wait_for_checkpoint_copies()
    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")
    return model
