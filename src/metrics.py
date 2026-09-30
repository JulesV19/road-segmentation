import torch


NUM_CLASSES = 8
IGNORE_INDEX = 255

CLASS_NAMES = [
    "void",         # 0 — output channel only, never a target after remapping (void → 255)
    "flat",
    "construction",
    "object",
    "nature",
    "sky",
    "human",
    "vehicle",
]


class SegMetric:
    """
    Standard (Cityscapes-style) mIoU: a confusion matrix is accumulated over every
    pixel of the split, and IoU_c = TP / (TP + FP + FN) is computed once at the end.

    This is NOT the same as averaging a per-image or per-batch mIoU, which weights
    small images/batches equally with large ones and gives non-comparable numbers.

    Class 0 (void) is excluded from the mean: it never appears in targets, but a
    valid pixel predicted as void still counts as a false negative for its true class.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        device: str | torch.device = "cpu",
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confmat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds:   (N, H, W) — argmax of logits (long)
            targets: (N, H, W) — ground truth labels (long), ignore_index allowed
        """
        valid = targets != self.ignore_index
        idx = targets[valid] * self.num_classes + preds[valid]
        self.confmat += torch.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def per_class_iou(self) -> list[float]:
        """IoU for classes 1..num_classes-1 (NaN if a class is absent from preds and targets)."""
        cm = self.confmat.cpu().double()  # float64 is not supported on MPS
        tp = cm.diag()
        union = cm.sum(dim=0) + cm.sum(dim=1) - tp
        iou = tp / union
        return iou[1:].tolist()

    def compute(self) -> tuple[float, list[float]]:
        per_class = self.per_class_iou()
        return torch.tensor(per_class, dtype=torch.float64).nanmean().item(), per_class

    def pixel_accuracy(self) -> float:
        cm = self.confmat.cpu().double()  # float64 is not supported on MPS
        return (cm.diag().sum() / cm.sum()).item()
