import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import evaluate

CFG = {"model": {"num_classes": 3}, "training": {"ignore_index": 255}}


def test_per_city_matches_hand_computed_iou():
    targets = torch.tensor([
        [[1, 1, 2, 2]],
        [[1, 1, 2, 2]],
        [[1, 2, 2, 2]],
    ])
    preds = torch.tensor([
        [[1, 1, 2, 2]],
        [[1, 2, 2, 2]],
        [[1, 1, 1, 2]],
    ])
    logits = torch.nn.functional.one_hot(preds, 3).permute(0, 3, 1, 2).float()
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)

    results = evaluate(torch.nn.Identity(), loader, CFG, "cpu", cities=["a", "a", "b"])

    assert results["per_city"]["a"]["num_images"] == 2
    assert results["per_city"]["a"]["per_class_iou"] == {"flat": 3 / 4, "construction": 4 / 5}
    assert results["per_city"]["b"]["per_class_iou"] == {"flat": 1 / 3, "construction": 1 / 3}
    assert "per_city" not in evaluate(torch.nn.Identity(), loader, CFG, "cpu")
