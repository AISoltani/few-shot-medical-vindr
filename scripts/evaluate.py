#!/usr/bin/env python
from __future__ import annotations
import argparse, json
import torch
from torch.utils.data import DataLoader
from medical_fsod.dataset import CSVBBoxDataset, collate_fn
from medical_fsod.model import build_fasterrcnn
from medical_fsod.eval import eval_iou50_f1
from medical_fsod.utils import load_classes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ann_csv", required=True)
    ap.add_argument("--classes_json", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score_thr", type=float, default=0.05)
    args=ap.parse_args()

    class_to_id=load_classes(args.classes_json)
    id_to_class={int(v):k for k,v in class_to_id.items()}
    id_to_class[0]="__background__"
    num_classes=max(id_to_class.keys())+1

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=CSVBBoxDataset(args.images_dir, args.ann_csv, class_to_id)
    loader=DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model=build_fasterrcnn(num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    model.to(device)

    metrics=eval_iou50_f1(model, loader, device, id_to_class, iou_thr=args.iou, score_thr=args.score_thr)
    print(json.dumps(metrics, indent=2))

if __name__=="__main__":
    main()
