import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_DENSE_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class DenseLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_a = nn.Parameter(
            torch.empty(
                rank,
                base_layer.in_features,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                rank,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return self.base_layer(x) + (lora_out * self.scaling)


def _matches_target(name: str, target_keywords: Iterable[str]) -> bool:
    return "language_model.layers" in name and any(keyword in name for keyword in target_keywords)


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, child_name, new_module)


def inject_dense_lora(
    model,
    rank: int,
    alpha: int,
    dropout: float,
    target_keywords: Iterable[str] = DEFAULT_DENSE_LORA_TARGETS,
):
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches_target(name, target_keywords)
    ]

    for name, module in replacements:
        _replace_module(model, name, DenseLoRALinear(module, rank, alpha, dropout))

    return model


def get_trainable_params(model) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count

    pct = 100 * trainable / total if total else 0
    print(f"trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    return trainable, total
