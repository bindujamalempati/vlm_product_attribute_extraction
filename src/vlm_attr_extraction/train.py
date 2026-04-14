from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from vlm_attr_extraction.data.dataset import (
    MultimodalAttributeDataset,
    build_text_vocab,
    AttributeEncoder,
    collate_fn,
)
from vlm_attr_extraction.models.model import MultimodalAttributeModel
from vlm_attr_extraction.utils.io import load_yaml, read_jsonl, write_json

try:
    import mlflow
except Exception:
    mlflow = None

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate(model, loader, device, attr_names):
    model.eval()
    y_true = {a: [] for a in attr_names}
    y_pred = {a: [] for a in attr_names}
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            text_ids = batch["text_ids"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            outputs = model(images, text_ids)
            for attr in attr_names:
                preds = outputs[attr].argmax(dim=1)
                y_true[attr].extend(labels[attr].cpu().tolist())
                y_pred[attr].extend(preds.cpu().tolist())
    metrics = {}
    f1s = []
    for attr in attr_names:
        f1 = f1_score(y_true[attr], y_pred[attr], average="macro", zero_division=0)
        metrics[f"{attr}_f1"] = float(f1)
        f1s.append(f1)
    metrics["macro_f1"] = float(sum(f1s) / len(f1s)) if f1s else 0.0
    return metrics

def main(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])

    train_records = read_jsonl(cfg["train_annotations"])
    val_records = read_jsonl(cfg["val_annotations"])
    all_records = train_records + val_records

    vocab = build_text_vocab(all_records)
    attr_encoder = AttributeEncoder(all_records, cfg["attributes"])
    attr_cardinalities = {a: len(attr_encoder.value_maps[a]) for a in cfg["attributes"]}

    train_ds = MultimodalAttributeDataset(
        annotations_path=cfg["train_annotations"],
        image_root=cfg["image_root"],
        vocab=vocab,
        attr_encoder=attr_encoder,
        max_text_len=cfg["max_text_len"],
        image_size=cfg["image_size"],
    )
    val_ds = MultimodalAttributeDataset(
        annotations_path=cfg["val_annotations"],
        image_root=cfg["image_root"],
        vocab=vocab,
        attr_encoder=attr_encoder,
        max_text_len=cfg["max_text_len"],
        image_size=cfg["image_size"],
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalAttributeModel(
        vocab_size=len(vocab.token_to_idx),
        attribute_cardinalities=attr_cardinalities,
        hidden_dim=cfg["hidden_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    use_mlflow = bool(cfg.get("mlflow", {}).get("enabled", False)) and mlflow is not None
    if use_mlflow:
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        mlflow.start_run()

    best_f1 = -1.0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            images = batch["images"].to(device)
            text_ids = batch["text_ids"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            outputs = model(images, text_ids)

            loss = 0.0
            for attr in cfg["attributes"]:
                loss = loss + criterion(outputs[attr], labels[attr])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        metrics = evaluate(model, val_loader, device, cfg["attributes"])
        metrics["epoch"] = epoch
        metrics["train_loss"] = epoch_loss / max(1, len(train_loader))
        history.append(metrics)

        if use_mlflow:
            mlflow.log_metrics(metrics, step=epoch)

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab.token_to_idx,
                    "attribute_value_maps": attr_encoder.value_maps,
                    "config": cfg,
                },
                output_dir / "best_model.pt",
            )

    write_json(output_dir / "metrics.json", history)
    if use_mlflow:
        mlflow.end_run()
    print(f"Training complete. Best macro_f1={best_f1:.4f}. Model saved to {output_dir / 'best_model.pt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)
