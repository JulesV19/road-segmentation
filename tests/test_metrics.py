import math

import torch

from src.metrics import IGNORE_INDEX, SegMetric


def test_perfect_prediction():
    target = torch.tensor([[[1, 2], [3, IGNORE_INDEX]]])
    metric = SegMetric(num_classes=4)
    metric.update(target.clone().clamp(max=3), target)
    miou, per_class = metric.compute()
    assert miou == 1.0
    assert per_class == [1.0, 1.0, 1.0]


def test_ignore_index_is_excluded():
    target = torch.tensor([[[1, IGNORE_INDEX]]])
    pred = torch.tensor([[[1, 2]]])
    metric = SegMetric(num_classes=3)
    metric.update(pred, target)
    _, per_class = metric.compute()
    assert per_class[0] == 1.0
    assert math.isnan(per_class[1])


def test_hand_computed_iou():
    target = torch.tensor([[[1, 1, 1, 2, 2]]])
    pred = torch.tensor([[[1, 1, 2, 2, 1]]])
    metric = SegMetric(num_classes=3)
    metric.update(pred, target)
    miou, per_class = metric.compute()
    assert per_class == [0.5, 1 / 3]
    assert math.isclose(miou, (0.5 + 1 / 3) / 2)


def test_void_prediction_counts_as_false_negative():
    target = torch.tensor([[[1, 1]]])
    pred = torch.tensor([[[1, 0]]])
    metric = SegMetric(num_classes=2)
    metric.update(pred, target)
    miou, _ = metric.compute()
    assert miou == 0.5


def test_accumulates_over_batches_not_averaged():
    metric = SegMetric(num_classes=2)
    metric.update(torch.tensor([[[1, 1, 1]]]), torch.tensor([[[1, 1, 1]]]))
    metric.update(torch.tensor([[[0]]]), torch.tensor([[[1]]]))
    miou, _ = metric.compute()
    assert miou == 0.75
