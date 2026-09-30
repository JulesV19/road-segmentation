import shutil
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

CLASS_COLORS = np.array([
    [0,   0,   0],
    [128, 64,  128],
    [70,  70,  70],
    [153, 153, 153],
    [107, 142, 35],
    [70,  130, 180],
    [220, 20,  60],
    [0,   0,   142],
], dtype=np.uint8)

IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    lut = np.vstack([CLASS_COLORS, IGNORE_COLOR])
    safe = np.where(mask == 255, len(CLASS_COLORS), mask).astype(np.int32)
    return lut[safe].astype(np.uint8)


_copy_thread: threading.Thread | None = None


def wait_for_checkpoint_copies():
    global _copy_thread
    if _copy_thread is not None:
        _copy_thread.join()
        _copy_thread = None


def save_checkpoint(state: dict, path: str | Path, drive_paths: list[str | Path] = ()):
    global _copy_thread
    wait_for_checkpoint_copies()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

    def _copy():
        for dst in drive_paths:
            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)

    _copy_thread = threading.Thread(target=_copy)
    _copy_thread.start()


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler: torch.amp.GradScaler | None = None,
    device: str = "cuda",
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    for name, obj in (("scheduler", scheduler), ("scaler", scaler)):
        if obj is None:
            continue
        if name in ckpt:
            obj.load_state_dict(ckpt[name])
        else:
            print(f"  WARNING: checkpoint has no {name} state (old format) — {name} restarts from scratch")
    return ckpt


def denormalize(image: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    img = image.cpu().numpy()
    img = img * np.array(std)[:, None, None] + np.array(mean)[:, None, None]
    return img.clip(0, 1).transpose(1, 2, 0)


def visualize_predictions(
    images: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
    n: int = 4,
):
    n = min(n, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(18, 3 * n))
    if n == 1:
        axes = axes[None]

    for i in range(n):
        axes[i, 0].imshow(denormalize(images[i]))
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(mask_to_rgb(targets[i].cpu().numpy()))
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(mask_to_rgb(preds[i].cpu().numpy()))
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    return fig


def example_figure(
    image: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    class_names: list[str],
    alpha: float = 0.5,
    title: str | None = None,
):
    img = denormalize(image)
    pred_rgb = mask_to_rgb(pred.cpu().numpy())
    overlay = (1 - alpha) * img + alpha * pred_rgb / 255.0

    fig, axes = plt.subplots(1, 4, figsize=(24, 3.6))
    panels = [
        (img, "Image"),
        (mask_to_rgb(target.cpu().numpy()), "Ground truth"),
        (pred_rgb, "Prediction"),
        (overlay, f"Overlay (α={alpha})"),
    ]
    for ax, (data, name) in zip(axes, panels):
        ax.imshow(data)
        ax.set_title(name)
        ax.axis("off")

    handles = [
        Patch(color=CLASS_COLORS[i] / 255.0, label=name)
        for i, name in enumerate(class_names) if i > 0
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig
