from __future__ import annotations
from typing import Dict, List

import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).flatten(1)
        return self.proj(feat)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, out_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.proj = nn.Linear(emb_dim, out_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)
        pooled = emb.mean(dim=1)
        return self.proj(pooled)

class MultimodalAttributeModel(nn.Module):
    def __init__(self, vocab_size: int, attribute_cardinalities: Dict[str, int], hidden_dim: int = 128) -> None:
        super().__init__()
        self.image_encoder = SmallCNN(out_dim=hidden_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, emb_dim=64, out_dim=hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.heads = nn.ModuleDict({
            attr: nn.Linear(hidden_dim, cardinality)
            for attr, cardinality in attribute_cardinalities.items()
        })

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_feat = self.image_encoder(images)
        text_feat = self.text_encoder(text_ids)
        fused = self.fusion(torch.cat([image_feat, text_feat], dim=1))
        return {attr: head(fused) for attr, head in self.heads.items()}
