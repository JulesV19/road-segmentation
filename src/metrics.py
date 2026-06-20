import torch


NUM_CLASSES = 8
IGNORE_INDEX = 255

CLASS_NAMES = [
    "void",         # 0 — should not appear after remapping (all → 255)
    "flat",         # 1
    "construction", # 2
    "object",       # 3
    "nature",       # 4
    "sky",          # 5
    "human",        # 6
    "vehicle",      # 7
]


def compute_miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, list[float]]:
    """
    Args:
        preds:   (N, H, W) — argmax of logits (long)
        targets: (N, H, W) — ground truth labels (long)
    Returns:
        mean_iou: scalar tensor
        per_class_iou: list of floats (only for classes present in targets)
    """
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    ious = []
    per_class = []
    for cls in range(1, num_classes):  # skip class 0 (void, never present after remapping)
        pred_c = preds == cls
        target_c = targets == cls
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0:
            per_class.append(float("nan"))
            continue
        iou = intersection / union
        ious.append(iou)
        per_class.append(iou.item())

    mean_iou = torch.stack(ious).mean() if ious else torch.tensor(0.0)
    return mean_iou, per_class
