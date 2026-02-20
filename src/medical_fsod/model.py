
from __future__ import annotations
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_fasterrcnn(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def freeze_backbone(model: torch.nn.Module) -> None:
    for p in model.backbone.body.parameters():
        p.requires_grad = False
