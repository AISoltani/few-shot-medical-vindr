# few-shot-medical (VinDr-CXR)

Few-shot **medical object detection** starter repo built around **VinDr-CXR** (Chest X-ray bounding boxes).

This repo is designed to be easy to share:
- Convert VinDr → internal CSV format
- Base/Novel split + K-shot sampling
- Baseline detector: Torchvision **Faster R-CNN**
- Simple evaluation (IoU@0.5 precision/recall/F1)

> Dataset note: VinDr-CXR has license/terms; this repo does not redistribute it.

## Setup:

### pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### conda
```bash
conda env create -f environment.yml
conda activate fewshot-medical
```

## Put VinDr-CXR here

```
data/raw/vindr/
  images/
  annotations.csv
```

The converter expects at least:
`image_id, class_name (or label), x_min, y_min, x_max, y_max`

## Convert
```bash
python scripts/convert_vindr_cxr.py --vindr_dir data/raw/vindr --out_dir data/processed/vindr
```

## Make few-shot split (example)
```bash
python scripts/make_fewshot_splits.py \
  --data_dir data/processed/vindr \
  --out_dir data/processed/vindr_splits \
  --base "Aortic enlargement,Atelectasis,Cardiomegaly" \
  --novel "Pleural effusion,Pneumothorax" \
  --k 10 --seed 42
```

## Train base
```bash
python scripts/train_base.py \
  --images_dir data/processed/vindr/images \
  --ann_csv data/processed/vindr_splits/base_train.csv \
  --classes_json data/processed/vindr/classes.json \
  --out_dir outputs/base
```

## Fine-tune novel (K-shot)
```bash
python scripts/finetune_novel.py \
  --images_dir data/processed/vindr/images \
  --ann_csv data/processed/vindr_splits/novel_kshot.csv \
  --classes_json data/processed/vindr/classes.json \
  --ckpt outputs/base/model.pt \
  --out_dir outputs/novel_ft \
  --freeze_backbone
```

## Evaluate
```bash
python scripts/evaluate.py \
  --images_dir data/processed/vindr/images \
  --ann_csv data/processed/vindr_splits/val.csv \
  --classes_json data/processed/vindr/classes.json \
  --ckpt outputs/novel_ft/model.pt
```
