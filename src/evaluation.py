import torch
from tqdm import tqdm

from src.metrics import CLASS_NAMES, SegMetric


@torch.no_grad()
def evaluate(model, loader, cfg: dict, device: str, criterion=None, cities: list[str] | None = None) -> dict:
    model.eval()
    use_amp = cfg["training"]["mixed_precision"] and device == "cuda"

    def new_metric():
        return SegMetric(cfg["model"]["num_classes"], cfg["training"]["ignore_index"], device)

    metric = new_metric()
    city_metrics = {city: new_metric() for city in dict.fromkeys(cities or [])}
    total_loss = 0.0
    num_images = 0

    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            if criterion is not None:
                total_loss += criterion(logits, masks).item()
        preds = logits.argmax(dim=1)
        metric.update(preds, masks)
        if cities:
            for i in range(len(images)):
                city_metrics[cities[num_images + i]].update(preds[i], masks[i])
        num_images += len(images)

    miou, per_class = metric.compute()
    results = {
        "miou": miou,
        "per_class_iou": dict(zip(CLASS_NAMES[1:], per_class)),
        "pixel_acc": metric.pixel_accuracy(),
        "loss": total_loss / len(loader) if criterion is not None else None,
        "num_images": num_images,
    }
    if cities:
        results["per_city"] = {
            city: {"miou": m.compute()[0], "num_images": cities.count(city)}
            for city, m in city_metrics.items()
        }
    return results
