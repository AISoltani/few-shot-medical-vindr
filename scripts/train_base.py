#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from medical_fsod.dataset import CSVBBoxDataset, collate_fn
from medical_fsod.model import build_fasterrcnn
from medical_fsod.train import train_one_epoch, save_checkpoint
from medical_fsod.utils import load_classes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ann_csv", required=True)
    ap.add_argument("--classes_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_pretrained", action="store_true")
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    class_to_id=load_classes(args.classes_json)
    num_classes=max(class_to_id.values())+1

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=CSVBBoxDataset(args.images_dir, args.ann_csv, class_to_id)
    loader=DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    model=build_fasterrcnn(num_classes, pretrained=not args.no_pretrained).to(device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for ep in range(1, args.epochs+1):
        loss=train_one_epoch(model, opt, loader, device, ep)
        print(f"epoch={ep} loss={loss:.4f}")
        save_checkpoint(model, out/"model.pt")

    (out/"train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("Saved", out/"model.pt")

if __name__=="__main__":
    main()
