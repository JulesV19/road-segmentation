"""
Run once locally before uploading to Google Drive.

Usage:
    python scripts/preprocess.py \
        --data_root "/path/to/cityscapes" \
        --output_dir "/path/to/preprocessed" \
        --width 512 --height 256

Output structure:
    preprocessed/
    ├── images/{train,val}/{city}/*.png
    ├── masks/{train,val}/{city}/*.png
    ├── train.csv
    └── val.csv
"""

import argparse
import csv
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Cityscapes label ID → 8-class group
# 255 = ignore (void, unlabeled, rectification borders, etc.)
LUT = np.full(256, 255, dtype=np.uint8)
LUT[[7, 8, 9, 10]] = 1                          # flat
LUT[[11, 12, 13, 14, 15, 16]] = 2               # construction
LUT[[17, 18, 19, 20]] = 3                       # object
LUT[[21, 22]] = 4                               # nature
LUT[23] = 5                                     # sky
LUT[[24, 25]] = 6                               # human
LUT[[26, 27, 28, 29, 30, 31, 32, 33]] = 7      # vehicle

CLASS_NAMES = {
    1: "flat", 2: "construction", 3: "object",
    4: "nature", 5: "sky", 6: "human", 7: "vehicle",
}


def process_sample(args):
    img_src, mask_src, img_dst, mask_dst, width, height = args

    img_dst.parent.mkdir(parents=True, exist_ok=True)
    mask_dst.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_src).convert("RGB")
    img = img.resize((width, height), Image.BILINEAR)
    img.save(img_dst, optimize=True)

    mask = np.array(Image.open(mask_src))
    mask = LUT[mask]
    mask_pil = Image.fromarray(mask, mode="L")
    mask_pil = mask_pil.resize((width, height), Image.NEAREST)
    mask_pil.save(mask_dst, optimize=True)

    assert set(np.unique(np.array(mask_pil))).issubset(
        set(range(8)) | {255}
    ), f"Unexpected label values in {mask_dst}"

    return True


def collect_samples(data_root: Path, split: str):
    img_dir = data_root / "leftImg8bit_trainvaltest" / "leftImg8bit" / split
    mask_dir = data_root / "gtFine_trainvaltest" / "gtFine" / split

    samples = []
    for city_dir in sorted(img_dir.iterdir()):
        city = city_dir.name
        for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
            stem = img_path.stem.replace("_leftImg8bit", "")
            mask_path = mask_dir / city / f"{stem}_gtFine_labelIds.png"
            if not mask_path.exists():
                print(f"  WARNING: mask not found for {img_path.name}, skipping")
                continue
            samples.append((city, stem, img_path, mask_path))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root of Cityscapes dataset")
    parser.add_argument("--output_dir", required=True, help="Where to write preprocessed data")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--workers", type=int, default=cpu_count())
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        print(f"\nCollecting {split} samples...")
        samples = collect_samples(data_root, split)
        print(f"  Found {len(samples)} samples")

        tasks = []
        csv_rows = []
        for city, stem, img_src, mask_src in samples:
            rel_img = Path("images") / split / city / f"{stem}.png"
            rel_mask = Path("masks") / split / city / f"{stem}.png"
            tasks.append((
                img_src, mask_src,
                out_root / rel_img, out_root / rel_mask,
                args.width, args.height,
            ))
            csv_rows.append({
                "image_path": rel_img.as_posix(),
                "mask_path": rel_mask.as_posix(),
            })

        print(f"  Processing with {args.workers} workers...")
        with Pool(args.workers) as pool:
            list(tqdm(pool.imap(process_sample, tasks), total=len(tasks)))

        csv_path = out_root / f"{split}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "mask_path"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  Manifest saved: {csv_path}")

    print("\nDone. Verify class distribution with:")
    print("  python -c \"import numpy as np; from PIL import Image; "
          "m=np.array(Image.open('<any_mask>')); print(np.unique(m, return_counts=True))\"")


if __name__ == "__main__":
    main()
