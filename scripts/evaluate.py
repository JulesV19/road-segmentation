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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import CityscapesDataset, build_loaders
from src.evaluation import evaluate
from src.metrics import CLASS_NAMES
from src.model import load_model
from src.utils import auto_device, example_figure


def markdown_tables(results: dict) -> str:
    rows = ["| Class | IoU |", "|---|---|"]
    rows += [f"| {name} | {iou:.3f} |" for name, iou in results["per_class_iou"].items()]
    rows.append(f"| **mean** | **{results['miou']:.3f}** |")
    rows += ["", "| City | Images | mIoU |", "|---|---|---|"]
    rows += [f"| {city} | {r['num_images']} | {r['miou']:.3f} |" for city, r in results["per_city"].items()]
    return "\n".join(rows)


@torch.no_grad()
def save_examples(model, cfg, device, out_dir: Path, n: int = 8):
    ds = CityscapesDataset(cfg["data"]["root"], cfg["data"]["val_csv"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, idx in enumerate(np.linspace(0, len(ds) - 1, n).round().astype(int)):
        image, mask = ds[idx]
        pred = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu()
        name = Path(ds.df.iloc[idx]["image_path"]).stem
        fig = example_figure(image, mask, pred, CLASS_NAMES, title=f"val · {name}")
        fig.savefig(out_dir / f"example_{k:02d}.png", dpi=100)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--data_root",    required=True, help="Preprocessed data root (contains val.csv)")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--output_json",  default="results/metrics.json")
    parser.add_argument("--examples_dir", default=None, help="If set, also save example figures here")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["root"] = args.data_root
    device = auto_device()
    model, ckpt = load_model(args.checkpoint, cfg, device)

    _, val_loader = build_loaders(cfg)
    cities = [Path(p).parent.name for p in val_loader.dataset.df["image_path"]]
    results = evaluate(model, val_loader, cfg, device, cities=cities)
    results.update({"checkpoint": Path(args.checkpoint).name, "epoch": ckpt["epoch"] + 1})

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nVal mIoU: {results['miou']:.4f}  ·  pixel acc: {results['pixel_acc']:.4f}  "
          f"·  {results['num_images']} images  ·  saved {out}\n")
    print(markdown_tables(results))

    if args.examples_dir:
        save_examples(model, cfg, device, Path(args.examples_dir))


if __name__ == "__main__":
    main()
