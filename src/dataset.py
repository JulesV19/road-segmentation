from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def build_transforms(cfg, split: str) -> A.Compose:
    aug = cfg["augmentation"]
    if split == "train":
        transforms = [A.HorizontalFlip(p=aug["random_hflip_p"])]
        if aug.get("color_jitter"):
            transforms.append(
                A.ColorJitter(
                    brightness=aug["color_jitter_brightness"],
                    contrast=aug["color_jitter_contrast"],
                    saturation=aug["color_jitter_saturation"],
                    hue=aug["color_jitter_hue"],
                    p=0.5,
                )
            )
        return A.Compose(transforms)
    return A.Compose([])


class CityscapesDataset(Dataset):
    def __init__(self, root: str, csv_file: str, split: str = "train", cfg: dict = None):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / csv_file)
        self.transform = build_transforms(cfg, split) if cfg else A.Compose([])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = np.array(Image.open(self.root / row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(self.root / row["mask_path"]))  # uint8, already remapped

        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        image = TF.to_tensor(image)  # HWC uint8 → CHW float [0,1]
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(mask).long()
        return image, mask


def build_loaders(cfg: dict):
    data_cfg = cfg["data"]
    train_ds = CityscapesDataset(
        data_cfg["root"], data_cfg["train_csv"], split="train", cfg=cfg
    )
    val_ds = CityscapesDataset(
        data_cfg["root"], data_cfg["val_csv"], split="val", cfg=cfg
    )

    num_workers = data_cfg["num_workers"]
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=data_cfg["pin_memory"],
        persistent_workers=(num_workers > 0 and data_cfg.get("persistent_workers", True)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader
