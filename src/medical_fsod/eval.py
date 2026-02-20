
from __future__ import annotations
from collections import defaultdict
from typing import Dict
import torch
from tqdm import tqdm
from .utils import Box, iou

@torch.no_grad()
def eval_iou50_f1(model, loader, device, id_to_class: Dict[int,str], iou_thr=0.5, score_thr=0.05):
    model.eval()
    tp=defaultdict(int); fp=defaultdict(int); fn=defaultdict(int)

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images=[im.to(device) for im in images]
        outputs=model(images)
        for out, tgt in zip(outputs, targets):
            gt_boxes=tgt["boxes"].cpu().numpy().tolist()
            gt_labels=tgt["labels"].cpu().numpy().tolist()

            keep=[(b,l,s) for b,l,s in zip(out["boxes"].cpu().tolist(),
                                          out["labels"].cpu().tolist(),
                                          out["scores"].cpu().tolist()) if s>=score_thr]
            pred_boxes=[k[0] for k in keep]
            pred_labels=[k[1] for k in keep]

            matched=set()
            for pb,pl in zip(pred_boxes,pred_labels):
                best_i, best_j=0.0, None
                for j,(gb,gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if j in matched or gl!=pl: continue
                    val=iou(Box(*pb), Box(*gb))
                    if val>best_i: best_i, best_j=val, j
                if best_j is not None and best_i>=iou_thr:
                    tp[pl]+=1; matched.add(best_j)
                else:
                    fp[pl]+=1
            for j,gl in enumerate(gt_labels):
                if j not in matched: fn[gl]+=1

    per={}
    for cid,name in id_to_class.items():
        if cid==0: continue
        T,Fp,Fn=tp[cid],fp[cid],fn[cid]
        p=T/(T+Fp) if (T+Fp)>0 else 0.0
        r=T/(T+Fn) if (T+Fn)>0 else 0.0
        f1=(2*p*r/(p+r)) if (p+r)>0 else 0.0
        per[name]={"precision":p,"recall":r,"f1":f1,"tp":T,"fp":Fp,"fn":Fn}

    TP=sum(tp.values()); FP=sum(fp.values()); FN=sum(fn.values())
    mp=TP/(TP+FP) if (TP+FP)>0 else 0.0
    mr=TP/(TP+FN) if (TP+FN)>0 else 0.0
    mf1=(2*mp*mr/(mp+mr)) if (mp+mr)>0 else 0.0
    return {"micro":{"precision":mp,"recall":mr,"f1":mf1,"tp":TP,"fp":FP,"fn":FN},"per_class":per}
