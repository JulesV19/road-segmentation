import torch
from transformers import get_cosine_schedule_with_warmup

from src.train import EarlyStopping
from src.utils import load_checkpoint, save_checkpoint, wait_for_checkpoint_copies


def _setup():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=50)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, opt, sched, scaler


def test_resume_restores_scheduler_and_early_stopping(tmp_path):
    model, opt, sched, scaler = _setup()
    for _ in range(12):
        opt.step()
        sched.step()
    early = EarlyStopping(patience=10)
    early.step(0.8)
    early.step(0.7)

    state = {
        "epoch": 3,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "early_stop": early.state_dict(),
        "best_miou": 0.8,
        "cfg": {"model": {"encoder": "efficientnet-b4"}},
    }
    save_checkpoint(state, tmp_path / "local.pth", drive_paths=[tmp_path / "drive" / "last.pth"])
    wait_for_checkpoint_copies()

    model2, opt2, sched2, scaler2 = _setup()
    ckpt = load_checkpoint(tmp_path / "drive" / "last.pth", model2, opt2, sched2, scaler2, device="cpu")
    early2 = EarlyStopping(patience=10)
    early2.load_state_dict(ckpt["early_stop"])

    assert sched2.get_last_lr() == sched.get_last_lr()
    assert sched2.last_epoch == 12  # warmup is not restarted
    assert early2.counter == 1 and early2.best == 0.8
    assert ckpt["best_miou"] == 0.8
    assert ckpt["cfg"]["model"]["encoder"] == "efficientnet-b4"
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)
