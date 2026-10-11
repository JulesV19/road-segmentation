import torch
from tqdm import tqdm

from src.metrics import CLASS_NAMES, SegMetric


def check_not_stale(pred: torch.Tensor, prev: torch.Tensor | None):
    if prev is not None and pred.shape == prev.shape and torch.equal(pred, prev):
        raise RuntimeError(
            "identical predictions for two different inputs — a GPU op most likely failed silently "
            "(e.g. MPS out of memory). Close other GPU jobs or rerun with --device cpu."
        )


@torch.no_grad()
def evaluate(model, loader, cfg: dict, device: str, criterion=None, cities: list[str] | None = None) -> dict:
    model.eval()
    use_amp = cfg["training"].get("mixed_precision", False) and device == "cuda"

    def new_metric():
        return SegMetric(
            num_classes=cfg["model"]["num_classes"],
            ignore_index=cfg["training"]["ignore_index"],
            device=device,
        )

    metric = new_metric()
    if cities is not None and len(cities) != len(loader.dataset):
        raise ValueError(f"got {len(cities)} city names for {len(loader.dataset)} samples")
    city_metrics = {city: new_metric() for city in dict.fromkeys(cities)} if cities else {}
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
        if cities:
            for i in range(images.shape[0]):
                city_metrics[cities[num_images + i]].update(preds[i], masks[i])
        num_images += images.shape[0]

    miou, per_class = metric.compute()
    results = {
        "miou": miou,
        "per_class_iou": dict(zip(CLASS_NAMES[1:], per_class)),
        "pixel_acc": metric.pixel_accuracy(),
        "loss": total_loss / len(loader) if criterion is not None else None,
        "num_images": num_images,
    }
    if cities:
        results["per_city"] = {}
        for city, m in city_metrics.items():
            city_miou, city_per_class = m.compute()
            results["per_city"][city] = {
                "miou": city_miou,
                "per_class_iou": dict(zip(CLASS_NAMES[1:], city_per_class)),
                "num_images": cities.count(city),
            }
    return results
