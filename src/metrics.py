import torch


NUM_CLASSES = 8
IGNORE_INDEX = 255

CLASS_NAMES = [
    "void",
    "flat",
    "construction",
    "object",
    "nature",
    "sky",
    "human",
    "vehicle",
]


class SegMetric:
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
        valid = targets != self.ignore_index
        idx = targets[valid] * self.num_classes + preds[valid]
        self.confmat += torch.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def per_class_iou(self) -> list[float]:
        cm = self.confmat.cpu().double()
        tp = cm.diag()
        union = cm.sum(dim=0) + cm.sum(dim=1) - tp
        iou = tp / union
        return iou[1:].tolist()

    def compute(self) -> tuple[float, list[float]]:
        per_class = self.per_class_iou()
        return torch.tensor(per_class, dtype=torch.float64).nanmean().item(), per_class

    def pixel_accuracy(self) -> float:
        cm = self.confmat.cpu().double()
        return (cm.diag().sum() / cm.sum()).item()
