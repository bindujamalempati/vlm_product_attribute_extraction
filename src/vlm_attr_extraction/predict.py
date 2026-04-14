from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
import numpy as np

from vlm_attr_extraction.data.dataset import VocabBundle, encode_text, simple_tokenize
from vlm_attr_extraction.models.model import MultimodalAttributeModel

def load_image(path: str, image_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

def main(model_path: str, image_path: str, text: str):
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

    image_tensor = load_image(image_path, config["image_size"])
    text_ids = encode_text(text, vocab, config["max_text_len"]).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor, text_ids)

    result = {}
    for attr, logits in outputs.items():
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        result[attr] = {
            "value": inverse_maps[attr][idx],
            "confidence": round(float(probs[idx].item()), 4),
        }
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    main(args.model_path, args.image_path, args.text)
