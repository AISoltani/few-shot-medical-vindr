
from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CSVBBoxDataset(Dataset):
    def __init__(self, images_dir: str | Path, ann_csv: str | Path, class_to_id: Dict[str,int], transforms=None):
        self.images_dir = Path(images_dir)
        self.class_to_id = class_to_id
        self.transforms = transforms
        df = pd.read_csv(ann_csv)

        if "class_name" not in df.columns and "label" in df.columns:
            df = df.rename(columns={"label":"class_name"})

        required = {"image_id","x_min","y_min","x_max","y_max","class_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {ann_csv}: {sorted(missing)}")

        df["image_id"] = df["image_id"].astype(str)
        self.df = df
        self.image_ids = sorted(df["image_id"].unique().tolist())

    def __len__(self): return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        rows = self.df[self.df["image_id"] == image_id]

        if "path" in rows.columns and rows["path"].notna().any():
            img_path = Path(rows["path"].iloc[0])
            if not img_path.is_absolute():
                img_path = self.images_dir / img_path
        else:
            for ext in [".png",".jpg",".jpeg",".tif",".tiff"]:
                cand = self.images_dir / f"{image_id}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            else:
                raise FileNotFoundError(f"Image not found for image_id={image_id} in {self.images_dir}")

        img = Image.open(img_path).convert("RGB")

        boxes = torch.tensor(rows[["x_min","y_min","x_max","y_max"]].to_numpy(dtype=np.float32))
        labels = torch.tensor([self.class_to_id[c] for c in rows["class_name"].astype(str).tolist()], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))
