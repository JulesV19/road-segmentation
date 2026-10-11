import csv
import math
import shutil
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import build_loaders
from src.evaluation import evaluate
from src.metrics import CLASS_NAMES
from src.model import build_model


def build_criterion(cfg: dict, device: str):
    t = cfg["training"]
    weights = torch.tensor(t["class_weights"], dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=weights, ignore_index=t["ignore_index"])
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=t["ignore_index"])

    def criterion(logits, targets):
        return t["ce_weight"] * ce(logits, targets) + t["dice_weight"] * dice(logits, targets)

    return criterion


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_factor(step):
        if step < warmup_steps:
            return step / warmup_steps  # linear warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay to 0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, cfg, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  train", leave=False)
    for step, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=cfg["training"]["mixed_precision"]):
            loss = criterion(model(images), masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

    return total_loss / len(loader)


def append_history(path: Path, row: dict):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def train(cfg: dict, resume_from: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = cfg["training"]

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, t["warmup_epochs"] * steps_per_epoch, t["epochs"] * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=t["mixed_precision"])

    ckpt_dir = Path(cfg["checkpointing"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_miou = 0, 0.0

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        for name, obj in [("model", model), ("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)]:
            obj.load_state_dict(ckpt[name])
        start_epoch, best_miou = ckpt["epoch"] + 1, ckpt["best_miou"]

    for epoch in range(start_epoch, t["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, cfg, device)
        val = evaluate(model, val_loader, cfg, device, criterion=criterion)
        is_best = val["miou"] > best_miou
        best_miou = max(best_miou, val["miou"])
        print(f"Epoch {epoch + 1}/{t['epochs']}  train_loss={train_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  val_mIoU={val['miou']:.4f}{'  (best)' if is_best else ''}")

        append_history(ckpt_dir / "history.csv", {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_miou": val["miou"],
            **{f"iou_{name}": val["per_class_iou"][name] for name in CLASS_NAMES[1:]},
        })

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
        }, ckpt_dir / "last.pth")
        if is_best:
            shutil.copy(ckpt_dir / "last.pth", ckpt_dir / "best.pth")

    print(f"Training complete. Best val mIoU: {best_miou:.4f}")
    return model
