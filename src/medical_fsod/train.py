
from __future__ import annotations
from pathlib import Path
import torch
from tqdm import tqdm

def train_one_epoch(model, optimizer, loader, device, epoch: int):
    model.train()
    total, n = 0.0, 0
    for images, targets in tqdm(loader, desc=f"train {epoch}", leave=False):
        images = [im.to(device) for im in images]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item()); n += 1
    return total / max(1,n)

@torch.no_grad()
def save_checkpoint(model, out_path: str | Path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
