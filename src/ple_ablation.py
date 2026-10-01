"""Gemma 4 PLE ablations: keep the branch frozen, or disable it entirely."""

import torch


def freeze_ple(model):
    """Freeze PLE-only weights while the branch keeps feeding every decoder layer."""
    language_model = model.model.language_model
    if not language_model.config.hidden_size_per_layer_input:
        raise ValueError("This model has no PLE branch to freeze")

    for name in ("embed_tokens_per_layer", "per_layer_model_projection", "per_layer_projection_norm"):
        getattr(language_model, name).requires_grad_(False)

    for layer in language_model.layers:
        for name in ("per_layer_input_gate", "per_layer_projection", "post_per_layer_input_norm"):
            getattr(layer, name).requires_grad_(False)

    return model


def disable_ple(model):
    """Feed zero PLE vectors into every text decoder layer.

    A zero vector makes the gated projection and its RMSNorm output zero, so
    each layer keeps its ordinary residual stream. This avoids changing the
    pretrained checkpoint's tensor shapes or saved architecture config.
    """
    freeze_ple(model)

    def zero_per_layer_input(_module, args, kwargs):
        if len(args) < 2 or args[1] is None:
            raise RuntimeError("Expected Gemma 4 decoder's positional PLE input")
        return (args[0], torch.zeros_like(args[1]), *args[2:]), kwargs

    for layer in model.model.language_model.layers:
        layer.register_forward_pre_hook(zero_per_layer_input, with_kwargs=True)

    # Save this flag with full-FT checkpoints; test_set.py reinstalls the hooks.
    model.config.ple_enabled = False
    return model
