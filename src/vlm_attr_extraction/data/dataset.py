from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm_attr_extraction.utils.io import read_jsonl

@dataclass
class VocabBundle:
    token_to_idx: Dict[str, int]
    idx_to_token: Dict[int, str]

def simple_tokenize(text: str) -> List[str]:
    return text.lower().replace("/", " ").replace("-", " ").replace(",", " ").replace(".", " ").split()

def build_text_vocab(records: List[dict], min_freq: int = 1) -> VocabBundle:
    freq: Dict[str, int] = {"<pad>": 10**9, "<unk>": 10**9}
    for rec in records:
        text = f"{rec.get('title', '')} {rec.get('description', '')}"
        for tok in simple_tokenize(text):
            freq[tok] = freq.get(tok, 0) + 1
    tokens = [tok for tok, c in freq.items() if c >= min_freq]
    token_to_idx = {tok: i for i, tok in enumerate(tokens)}
    idx_to_token = {i: tok for tok, i in token_to_idx.items()}
    return VocabBundle(token_to_idx=token_to_idx, idx_to_token=idx_to_token)

def encode_text(text: str, vocab: VocabBundle, max_len: int) -> torch.Tensor:
    toks = simple_tokenize(text)[:max_len]
    ids = [vocab.token_to_idx.get(tok, vocab.token_to_idx["<unk>"]) for tok in toks]
    ids += [vocab.token_to_idx["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

class AttributeEncoder:
    def __init__(self, records: List[dict], attribute_names: List[str]):
        self.attribute_names = attribute_names
        self.value_maps: Dict[str, Dict[str, int]] = {}
        self.inverse_maps: Dict[str, Dict[int, str]] = {}
        for attr in attribute_names:
            values = sorted({
                str(rec.get("attributes", {}).get(attr, "unknown"))
                for rec in records
            } | {"unknown"})
            self.value_maps[attr] = {v: i for i, v in enumerate(values)}
            self.inverse_maps[attr] = {i: v for v, i in self.value_maps[attr].items()}

    def encode(self, attr_dict: Dict[str, str]) -> Dict[str, int]:
        out = {}
        for attr in self.attribute_names:
            val = str(attr_dict.get(attr, "unknown"))
            out[attr] = self.value_maps[attr].get(val, self.value_maps[attr]["unknown"])
        return out

class MultimodalAttributeDataset(Dataset):
    def __init__(
        self,
        annotations_path: str,
        image_root: str,
        vocab: VocabBundle,
        attr_encoder: AttributeEncoder,
        max_text_len: int = 96,
        image_size: int = 224,
    ) -> None:
        self.records = read_jsonl(annotations_path)
        self.image_root = Path(image_root)
        self.vocab = vocab
        self.attr_encoder = attr_encoder
        self.max_text_len = max_text_len
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_name: str) -> torch.Tensor:
        path = self.image_root / image_name
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(img).astype("float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image_tensor = self._load_image(rec["image"])
        text = f"{rec.get('title', '')} {rec.get('description', '')}"
        text_ids = encode_text(text, self.vocab, self.max_text_len)
        labels = self.attr_encoder.encode(rec.get("attributes", {}))
        return {
            "id": rec["id"],
            "image": image_tensor,
            "text_ids": text_ids,
            "labels": labels,
            "raw_text": text,
            "image_name": rec["image"],
        }

def collate_fn(batch: List[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch], dim=0)
    text_ids = torch.stack([item["text_ids"] for item in batch], dim=0)
    ids = [item["id"] for item in batch]
    raw_text = [item["raw_text"] for item in batch]
    image_names = [item["image_name"] for item in batch]
    label_keys = batch[0]["labels"].keys()
    labels = {k: torch.tensor([item["labels"][k] for item in batch], dtype=torch.long) for k in label_keys}
    return {
        "ids": ids,
        "images": images,
        "text_ids": text_ids,
        "labels": labels,
        "raw_text": raw_text,
        "image_names": image_names,
    }
