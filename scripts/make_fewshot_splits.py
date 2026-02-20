#!/usr/bin/env python
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--novel", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir/"annotations.csv")
    df["image_id"] = df["image_id"].astype(str)

    base = [c.strip() for c in args.base.split(",") if c.strip()]
    novel = [c.strip() for c in args.novel.split(",") if c.strip()]
    if set(base) & set(novel):
        raise ValueError("Base and novel classes overlap.")

    all_classes = set(df["class_name"].astype(str).unique().tolist())
    for c in base+novel:
        if c not in all_classes:
            raise ValueError(f"Class not found: {c}")

    rng = random.Random(args.seed)
    image_ids = sorted(df["image_id"].unique().tolist())
    rng.shuffle(image_ids)
    n_val = max(1, int(len(image_ids)*args.val_frac))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    train_df = df[df["image_id"].isin(train_ids)].copy()
    val_df = df[df["image_id"].isin(val_ids)].copy()

    base_train = train_df[train_df["class_name"].isin(base)].copy()
    base_train.to_csv(out_dir/"base_train.csv", index=False)

    novel_train = train_df[train_df["class_name"].isin(novel)].copy()
    parts=[]
    for cls in novel:
        sub = novel_train[novel_train["class_name"]==cls]
        imgs = sorted(sub["image_id"].unique().tolist())
        rng2 = random.Random(args.seed + (abs(hash(cls))%10000))
        rng2.shuffle(imgs)
        chosen = set(imgs[:min(args.k, len(imgs))])
        parts.append(sub[sub["image_id"].isin(chosen)])
    novel_kshot = pd.concat(parts, axis=0).reset_index(drop=True)
    novel_kshot.to_csv(out_dir/"novel_kshot.csv", index=False)

    val_df.to_csv(out_dir/"val.csv", index=False)

    meta = {"base_classes":base,"novel_classes":novel,"k":args.k,"seed":args.seed,"val_frac":args.val_frac}
    (out_dir/"split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote splits to", out_dir)

if __name__ == "__main__":
    main()
