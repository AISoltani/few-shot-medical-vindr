#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import pandas as pd
from medical_fsod.utils import save_classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vindr_dir", required=True, help="data/raw/vindr (images/ + annotations.csv)")
    ap.add_argument("--out_dir", required=True, help="data/processed/vindr")
    ap.add_argument("--img_subdir", default="images")
    ap.add_argument("--ann_name", default="annotations.csv")
    args = ap.parse_args()

    vindr = Path(args.vindr_dir)
    out = Path(args.out_dir)
    (out/"images").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(vindr/args.ann_name)

    if "class_name" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label":"class_name"})
    if "x_min" not in df.columns and "xmin" in df.columns:
        df = df.rename(columns={"xmin":"x_min","ymin":"y_min","xmax":"x_max","ymax":"y_max"})

    req = {"image_id","class_name","x_min","y_min","x_max","y_max"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")

    df["image_id"] = df["image_id"].astype(str)

    classes = sorted(df["class_name"].astype(str).unique().tolist())
    class_to_id = {c:i+1 for i,c in enumerate(classes)}  # 0 is background
    save_classes(class_to_id, out/"classes.json")

    df.to_csv(out/"annotations.csv", index=False)

    src = vindr/args.img_subdir
    exts = {".png",".jpg",".jpeg",".tif",".tiff"}
    copied = 0
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            shutil.copy2(p, out/"images"/p.name)
            copied += 1
    print(f"Saved {out/'annotations.csv'} and copied {copied} images.")

if __name__ == "__main__":
    main()
