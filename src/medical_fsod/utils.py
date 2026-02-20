
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import random
from typing import Dict

def set_seed(seed: int) -> None:
    random.seed(seed)

def load_classes(classes_json: str | Path) -> Dict[str, int]:
    with open(classes_json, "r", encoding="utf-8") as f:
        return json.load(f)

def save_classes(mapping: Dict[str, int], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

@dataclass
class Box:
    x1: float; y1: float; x2: float; y2: float

def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1,b.x1), max(a.y1,b.y1)
    ix2, iy2 = min(a.x2,b.x2), min(a.y2,b.y2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    area_a = max(0.0,a.x2-a.x1)*max(0.0,a.y2-a.y1)
    area_b = max(0.0,b.x2-b.x1)*max(0.0,b.y2-b.y1)
    union = area_a + area_b - inter
    return float(inter/union) if union>0 else 0.0
