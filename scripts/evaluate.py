"""
Evaluate a checkpoint on the Cityscapes val split with the standard (dataset-level) mIoU.

Usage:
    python scripts/evaluate.py \
        --checkpoint path/to/best.pth \
        --data_root  path/to/preprocessed \
        [--output_json results/metrics.json] \
        [--examples_dir assets/examples --n_examples 8] \
        [--device cuda|mps|cpu] [--encoder efficientnet-b4] [--skip_eval]

Writes a JSON with the metrics and prints a Markdown table ready for the README.
Example figures use evenly spaced val indices (not hand-picked).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import CityscapesDataset, build_loaders
from src.evaluation import check_not_stale, evaluate
from src.metrics import CLASS_NAMES
from src.model import load_model_for_inference
from src.utils import example_figure


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def markdown_table(results: dict) -> str:
    rows = ["| Class | IoU |", "|---|---|"]
    rows += [f"| {name} | {iou:.3f} |" for name, iou in results["per_class_iou"].items()]
    rows.append(f"| **mean** | **{results['miou']:.3f}** |")
    return "\n".join(rows)


@torch.no_grad()
def save_examples(model, cfg, device, out_dir: Path, n: int):
    ds = CityscapesDataset(cfg["data"]["root"], cfg["data"]["val_csv"], split="val")
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(ds) - 1, n).round().astype(int)
    prev = None
    for k, idx in enumerate(indices):
        image, mask = ds[idx]
        logits = model(image.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1)[0].cpu()
        check_not_stale(pred, prev)
        prev = pred
        name = Path(ds.df.iloc[idx]["image_path"]).stem
        fig = example_figure(image, mask, pred, CLASS_NAMES, alpha=0.5, title=f"val · {name}")
        path = out_dir / f"example_{k:02d}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  saved {path}  ({name})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--data_root",    required=True, help="Preprocessed data root (contains val.csv)")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--device",       default=None)
    parser.add_argument("--encoder",      default=None,
                        help="Override encoder (only needed for old checkpoints that do not embed their config)")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--num_workers",  type=int, default=2)
    parser.add_argument("--output_json",  default="results/metrics.json")
    parser.add_argument("--examples_dir", default=None, help="If set, write example figures here")
    parser.add_argument("--n_examples",   type=int, default=8)
    parser.add_argument("--skip_eval",    action="store_true",
                        help="Only write example figures (requires --examples_dir)")
    args = parser.parse_args()

    device = args.device or auto_device()
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["root"] = args.data_root
    cfg["data"]["num_workers"] = args.num_workers
    cfg["data"]["pin_memory"] = device == "cuda"
    cfg["training"]["batch_size"] = args.batch_size

    model, ckpt = load_model_for_inference(args.checkpoint, cfg, device, encoder=args.encoder)

    if args.skip_eval:
        if not args.examples_dir:
            parser.error("--skip_eval requires --examples_dir")
        save_examples(model, cfg, device, Path(args.examples_dir), args.n_examples)
        return

    _, val_loader = build_loaders(cfg)

    results = evaluate(model, val_loader, cfg, device)
    first_image = val_loader.dataset.root / val_loader.dataset.df.iloc[0]["image_path"]
    width, height = Image.open(first_image).size
    results.update({
        "checkpoint": Path(args.checkpoint).name,
        "epoch": ckpt.get("epoch", -1) + 1,
        "encoder": args.encoder or (ckpt["cfg"] if "cfg" in ckpt else cfg)["model"]["encoder"],
        "eval_resolution": f"{width}x{height}",
        "metric": "dataset-level mIoU over 7 Cityscapes categories (confusion matrix, void excluded)",
    })

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nVal mIoU: {results['miou']:.4f}  ·  pixel acc: {results['pixel_acc']:.4f}  "
          f"·  {results['num_images']} images")
    print(f"Saved {out}\n")
    print(markdown_table(results))

    if args.examples_dir:
        print()
        if device == "mps":
            torch.mps.empty_cache()  # release the eval batches' memory before the example passes
        save_examples(model, cfg, device, Path(args.examples_dir), args.n_examples)


if __name__ == "__main__":
    main()
