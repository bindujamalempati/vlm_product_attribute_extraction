from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
import numpy as np

from vlm_attr_extraction.data.dataset import VocabBundle, encode_text
from vlm_attr_extraction.models.model import MultimodalAttributeModel

APP_STATE = {}
app = FastAPI(title="VLM Product Attribute Extraction API")

def _load_image(path: str, image_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

def load_model(model_path: str = "outputs/run_01/best_model.pt"):
    if APP_STATE.get("loaded"):
        return
    ckpt = torch.load(model_path, map_location="cpu")
    vocab = VocabBundle(
        token_to_idx=ckpt["vocab"],
        idx_to_token={i: t for t, i in ckpt["vocab"].items()},
    )
    value_maps = ckpt["attribute_value_maps"]
    inverse_maps = {attr: {idx: val for val, idx in vmap.items()} for attr, vmap in value_maps.items()}
    config = ckpt["config"]
    attr_cardinalities = {a: len(value_maps[a]) for a in value_maps}
    model = MultimodalAttributeModel(
        vocab_size=len(vocab.token_to_idx),
        attribute_cardinalities=attr_cardinalities,
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    APP_STATE["loaded"] = True
    APP_STATE["model"] = model
    APP_STATE["vocab"] = vocab
    APP_STATE["inverse_maps"] = inverse_maps
    APP_STATE["config"] = config

@app.on_event("startup")
def startup_event():
    model_path = Path("outputs/run_01/best_model.pt")
    if model_path.exists():
        load_model(str(model_path))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    title: str = Form(...),
    description: str = Form(""),
    image: UploadFile = File(...),
):
    if not APP_STATE.get("loaded"):
        return {"error": "Model not loaded. Train first or place a checkpoint at outputs/run_01/best_model.pt"}
    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    config = APP_STATE["config"]
    image_tensor = _load_image(tmp_path, config["image_size"])
    text = f"{title} {description}".strip()
    text_ids = encode_text(text, APP_STATE["vocab"], config["max_text_len"]).unsqueeze(0)

    with torch.no_grad():
        outputs = APP_STATE["model"](image_tensor, text_ids)

    result = {}
    for attr, logits in outputs.items():
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        result[attr] = {
            "value": APP_STATE["inverse_maps"][attr][idx],
            "confidence": round(float(probs[idx].item()), 4),
        }
    return result
