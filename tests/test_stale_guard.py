import pytest
import torch

from src.evaluation import check_not_stale


def test_identical_predictions_raise():
    pred = torch.randint(0, 8, (2, 4, 4))
    with pytest.raises(RuntimeError, match="failed silently"):
        check_not_stale(pred, pred.clone())


def test_different_predictions_pass():
    a = torch.zeros(2, 4, 4, dtype=torch.long)
    b = a.clone()
    b[0, 0, 0] = 1
    check_not_stale(a, None)
    check_not_stale(b, a)
    check_not_stale(a[:1], a)  # last batch can be smaller
