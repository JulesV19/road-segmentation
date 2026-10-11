import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    # ignored pixels (255) are drawn black, like the unused void class 0
    return CLASS_COLORS[np.where(mask == 255, 0, mask)]


def denormalize(image: torch.Tensor) -> np.ndarray:
    img = image.cpu().numpy().transpose(1, 2, 0)
    return (img * IMAGENET_STD + np.array(IMAGENET_MEAN)).clip(0, 1)


def example_figure(image, target, pred, class_names, alpha=0.5, title=None):
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
