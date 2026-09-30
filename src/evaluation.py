import torch
from tqdm import tqdm

from src.metrics import CLASS_NAMES, SegMetric


def check_not_stale(pred: torch.Tensor, prev: torch.Tensor | None):
    """
    On MPS, a command buffer that runs out of memory only logs an error and the
    output tensor keeps its previous content. Two different inputs never give a
    byte-identical argmax map, so treat that as a failed forward pass.
    """
    if prev is not None and pred.shape == prev.shape and torch.equal(pred, prev):
        raise RuntimeError(
            "identical predictions for two different inputs — a GPU op most likely failed silently "
            "(e.g. MPS out of memory). Close other GPU jobs or rerun with --device cpu."
        )


@torch.no_grad()
def evaluate(model, loader, cfg: dict, device: str, criterion=None) -> dict:
    """
    Run the model over a full split and return split-level metrics.

    Returns:
        {"miou": float, "per_class_iou": {name: float}, "pixel_acc": float,
         "loss": float | None, "num_images": int}
    """
    model.eval()
    use_amp = cfg["training"].get("mixed_precision", False) and device == "cuda"
    metric = SegMetric(
        num_classes=cfg["model"]["num_classes"],
        ignore_index=cfg["training"]["ignore_index"],
        device=device,
    )
    total_loss = 0.0
    num_images = 0
    prev_preds = None

    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            if criterion is not None:
                total_loss += criterion(logits, masks).item()
        preds = logits.argmax(dim=1)
        check_not_stale(preds, prev_preds)
        prev_preds = preds
        metric.update(preds, masks)
        num_images += images.shape[0]

    miou, per_class = metric.compute()
    return {
        "miou": miou,
        "per_class_iou": dict(zip(CLASS_NAMES[1:], per_class)),
        "pixel_acc": metric.pixel_accuracy(),
        "loss": total_loss / len(loader) if criterion is not None else None,
        "num_images": num_images,
    }
