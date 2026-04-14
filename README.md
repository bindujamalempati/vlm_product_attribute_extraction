# Vision Language Model for Product Attribute Extraction

A complete GitHub-ready project for multimodal **product attribute extraction** using **image + text** inputs. This repo is designed to match a realistic secondhand marketplace use case where listings are noisy, descriptions are inconsistent, and structured attributes must be extracted from unstructured content.

## What is included

- Runnable multimodal training pipeline
- Small included sample dataset for immediate testing
- Real dataset download + preparation hooks
- Inference script
- FastAPI serving layer
- Dockerfile
- Kubernetes deployment manifest
- MLflow-ready training loop
- Clean project structure for GitHub

## Project summary

This project demonstrates a multimodal pipeline that combines:

- **Visual signal** from product images
- **Text signal** from titles and descriptions
- **Multi-head classification** for structured attribute prediction

Example supported attributes in the scaffold:
- category
- primary_color
- pattern
- sleeve_length
- neckline
- length
- material
- style
- closure
- condition
- height

## Real-world dataset strategy

This repo is built to extend to real public datasets such as:

- **DeepFashion / DeepFashion2**
- **Amazon Berkeley Objects (ABO)**

These datasets are **not bundled inside the zip** because they are large and have their own download terms. Instead, the repo includes:

- `scripts/download_real_datasets.py`
- `scripts/prepare_real_data.py`

That keeps the project honest, lightweight, and GitHub-safe.

## Quick start

### 1) Create environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
set PYTHONPATH=src
```

For macOS/Linux use:

```bash
export PYTHONPATH=src
```

### 2) Train on the included sample

```bash
python -m vlm_attr_extraction.train --config configs/train.yaml
```

This writes:
- `outputs/run_01/best_model.pt`
- `outputs/run_01/metrics.json`

### 3) Run prediction

```bash
python -m vlm_attr_extraction.predict ^
  --model-path outputs/run_01/best_model.pt ^
  --image-path data/sample/images/sample_006.jpg ^
  --text "Yellow polka dot blouse with short sleeves in good condition"
```

macOS/Linux:

```bash
python -m vlm_attr_extraction.predict \
  --model-path outputs/run_01/best_model.pt \
  --image-path data/sample/images/sample_006.jpg \
  --text "Yellow polka dot blouse with short sleeves in good condition"
```

### 4) Start API

```bash
uvicorn vlm_attr_extraction.api.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## API usage

`POST /predict`

Form fields:
- `title`
- `description`
- `image` (file)

Example response:

```json
{
  "category": {"value": "blouse", "confidence": 0.81},
  "primary_color": {"value": "yellow", "confidence": 0.76}
}
```

## Real dataset integration path

### Step 1
Run:

```bash
python scripts/download_real_datasets.py
```

This writes local instructions for placing DeepFashion2 and ABO under `data/raw/`.

### Step 2
Manually download the official dataset archives and unzip them to:

```text
data/raw/deepfashion2/
data/raw/abo/
```

### Step 3
Customize and extend:

```bash
python scripts/prepare_real_data.py
```

The project already defines the normalized output format expected by the trainer:

```json
{
  "id": "listing_123",
  "image": "images/item_123.jpg",
  "title": "Black leather biker jacket cropped fit",
  "description": "Secondhand black leather biker jacket with zip closure.",
  "attributes": {
    "category": "jacket",
    "primary_color": "black",
    "material": "leather",
    "style": "biker"
  }
}
```

## Model design

The included working model is intentionally lightweight so it runs locally:

- small CNN image encoder
- token embedding + mean pooling text encoder
- fusion MLP
- multi-head classification heads for each attribute

This is a **working baseline**. For a stronger production-grade extension, you can replace it with:
- CLIP vision encoder
- BERT / DistilBERT text encoder
- cross-attention fusion
- multi-task loss weighting
- calibration and confidence scoring

## Docker

```bash
docker build -t vlm-attribute-api -f docker/Dockerfile .
docker run -p 8000:8000 vlm-attribute-api
```

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

## Smoke test

```bash
python scripts/run_smoke_test.py
```

## Why this repo is realistic

- Uses real multimodal input structure
- Supports multi-attribute prediction
- Separates raw data, processed data, model code, serving, and infra
- Includes deployment artifacts
- Keeps dataset handling compliant and practical
- Can be shown on GitHub without pretending giant proprietary/public datasets are embedded

## Suggested next improvements

- add CLIP/Transformers backbone
- add dataset-specific mappers for DeepFashion2/ABO
- add SHAP or gradient-based image-text explanations
- add batch inference job
- add MLflow artifact logging
- add model evaluation dashboard

## Folder structure

```text
vlm_product_attribute_extraction/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── docker/
├── k8s/
├── scripts/
├── src/vlm_attr_extraction/
├── tests/
├── README.md
└── requirements.txt
```
