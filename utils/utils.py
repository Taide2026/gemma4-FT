import torch
from typing import List


def _log(*args):
    print("[Stage1]", *args, flush=True)


def _set_requires_grad(params, value: bool):
    for p in params:
        p.requires_grad = value


def _count_params(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    return trainable, total


def _print_trainable_parameters(model, max_names: int = 80):
    trainable, total = _count_params(model)
    ratio = 100 * trainable / total if total > 0 else 0.0

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

    _log(f"trainable params: {trainable:,} / {total:,} ({ratio:.4f}%)")
    _log(f"num trainable tensors: {len(trainable_names)}")
    _log(f"trainable modules / params first {max_names}:")

    for name in trainable_names[:max_names]:
        _log(f"  {name}")

    if len(trainable_names) > max_names:
        _log(f"  ... and {len(trainable_names) - max_names} more trainable tensors")


def _pad_sequence(sequences: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    """右側 padding，回傳 [batch, max_len] tensor。"""
    max_len = max(s.size(0) for s in sequences)
    batch = sequences[0].new_full((len(sequences), max_len), padding_value)

    for i, seq in enumerate(sequences):
        batch[i, :seq.size(0)] = seq

    return batch
