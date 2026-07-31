import torch
from typing import List, Iterable, Tuple, Optional


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


def _unwrap_model(model):
    """
    Handles PEFT / DDP / DeepSpeed-like wrappers.

    We do not assume a fixed model path such as:
        model.model.vision_tower

    because after PEFT wrapping the structure may change.
    """
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass

    if hasattr(model, "module"):
        return _unwrap_model(model.module)

    return model


def _get_module_by_path(model, path: str):
    cur = model
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _find_first_existing_module(model, candidate_paths: Iterable[str]):
    """
    Try exact attribute paths first.
    This is useful when we know common structures like:
        model.language_model
        model.model.language_model
        model.embed_vision
        model.model.embed_vision
    """
    base_model = _unwrap_model(model)

    roots = [model, base_model]

    for root in roots:
        for path in candidate_paths:
            module = _get_module_by_path(root, path)
            if module is not None:
                return path, module

    return None, None


def _find_modules_by_keywords(
    model,
    include_keywords: Iterable[str],
    exclude_keywords: Optional[Iterable[str]] = None,
) -> List[Tuple[str, torch.nn.Module]]:
    """
    Search named_modules() by keywords.

    Returns modules whose names contain one of include_keywords and
    do not contain any exclude_keywords.
    """
    base_model = _unwrap_model(model)

    include_keywords = [k.lower() for k in include_keywords]
    exclude_keywords = [k.lower() for k in (exclude_keywords or [])]

    matched = []

    for name, module in base_model.named_modules():
        lname = name.lower()

        if not name:
            continue

        if any(k in lname for k in include_keywords):
            if not any(k in lname for k in exclude_keywords):
                matched.append((name, module))

    return matched


def _select_top_level_modules(
    modules: List[Tuple[str, torch.nn.Module]]
) -> List[Tuple[str, torch.nn.Module]]:
    """
    named_modules() returns both parent and child modules.
    We usually only want the highest-level matched modules.

    Example:
        model.vision_model
        model.vision_model.encoder
        model.vision_model.encoder.layers.0

    We keep:
        model.vision_model
    """
    selected = []

    for name, module in modules:
        should_skip = False

        for selected_name, _ in selected:
            if name.startswith(selected_name + "."):
                should_skip = True
                break

        if not should_skip:
            selected.append((name, module))

    return selected


def _move_module_to_dtype_device(module, compute_dtype=None, device=None):
    kwargs = {}

    if compute_dtype is not None:
        kwargs["dtype"] = compute_dtype

    if device is not None:
        kwargs["device"] = device

    if kwargs:
        module.to(**kwargs)


def _freeze_all(model):
    _set_requires_grad(model.parameters(), False)


def _freeze_llm(model):
    """
    Freeze the language model part.

    Try exact paths first. If not found, fall back to keyword search.
    """
    candidate_paths = [
        "language_model",
        "model.language_model",
        "base_model.model.language_model",
        "base_model.model.model.language_model",
    ]

    path, language_model = _find_first_existing_module(model, candidate_paths)

    if language_model is not None:
        _set_requires_grad(language_model.parameters(), False)
        _log(f"Frozen language model: {path}")
        return

    matched = _find_modules_by_keywords(
        model,
        include_keywords=["language_model"],
        exclude_keywords=[],
    )
    matched = _select_top_level_modules(matched)

    if not matched:
        _log("Warning: could not find language_model module. Freezing all parameters instead.")
        _freeze_all(model)
        return

    for name, module in matched:
        _set_requires_grad(module.parameters(), False)
        _log(f"Frozen language model module: {name}")


def _unfreeze_projector(model, compute_dtype=None, device=None):
    """
    Unfreeze multimodal projector.

    In your original code, Gemma 4 comment says:
        vision_tower: frozen
        embed_vision: projector, trained

    So we prioritize embed_vision first.
    """
    candidate_paths = [
        "embed_vision",
        "model.embed_vision",
        "multi_modal_projector",
        "model.multi_modal_projector",
        "mm_projector",
        "model.mm_projector",
        "visual_projection",
        "model.visual_projection",
        "vision_projector",
        "model.vision_projector",
        "image_projector",
        "model.image_projector",
    ]

    path, projector = _find_first_existing_module(model, candidate_paths)

    if projector is not None:
        _move_module_to_dtype_device(projector, compute_dtype, device)
        _set_requires_grad(projector.parameters(), True)
        _log(f"Unfrozen projector: {path} ({projector.__class__.__name__})")
        return

    matched = _find_modules_by_keywords(
        model,
        include_keywords=[
            "embed_vision",
            "projector",
            "multi_modal_projector",
            "mm_projector",
            "visual_projection",
            "vision_projector",
            "image_projector",
        ],
        exclude_keywords=[],
    )
    matched = _select_top_level_modules(matched)

    if not matched:
        _log_possible_multimodal_modules(model)
        raise AttributeError(
            "Could not find projector module. "
            "Please check printed module names and update projector keywords."
        )

    for name, module in matched:
        _move_module_to_dtype_device(module, compute_dtype, device)
        _set_requires_grad(module.parameters(), True)
        _log(f"Unfrozen projector module: {name} ({module.__class__.__name__})")


def _freeze_image_encoder(model):
    """
    Freeze image / vision encoder modules.
    """
    matched = _find_image_encoder_modules(model)

    if not matched:
        _log("Warning: could not find image encoder module to freeze.")
        return

    for name, module in matched:
        _set_requires_grad(module.parameters(), False)
        _log(f"Frozen image encoder module: {name} ({module.__class__.__name__})")


def _unfreeze_image_encoder(model, compute_dtype=None, device=None):
    """
    Unfreeze image / vision encoder modules.

    Do not hardcode:
        model.model.vision_tower

    because the actual Gemma 4 HF class may use another name.
    """
    matched = _find_image_encoder_modules(model)

    if not matched:
        _log_possible_multimodal_modules(model)
        raise AttributeError(
            "Could not find image encoder module. "
            "Please check printed module names and update image encoder keywords."
        )

    for name, module in matched:
        _move_module_to_dtype_device(module, compute_dtype, device)
        _set_requires_grad(module.parameters(), True)
        _log(f"Unfrozen image encoder module: {name} ({module.__class__.__name__})")


def _find_image_encoder_modules(model) -> List[Tuple[str, torch.nn.Module]]:
    """
    Find possible image / vision encoder modules.

    We intentionally exclude projector-like modules such as embed_vision,
    because projector should be controlled by _unfreeze_projector().
    """
    candidate_paths = [
        "vision_tower",
        "model.vision_tower",
        "vision_model",
        "model.vision_model",
        "vision_encoder",
        "model.vision_encoder",
        "visual_encoder",
        "model.visual_encoder",
        "image_encoder",
        "model.image_encoder",
        "image_tower",
        "model.image_tower",
    ]

    path, module = _find_first_existing_module(model, candidate_paths)

    if module is not None:
        return [(path, module)]

    matched = _find_modules_by_keywords(
        model,
        include_keywords=[
            "vision_tower",
            "vision_model",
            "vision_encoder",
            "visual_encoder",
            "image_encoder",
            "image_tower",
            "siglip",
            "clip",
        ],
        exclude_keywords=[
            "embed_vision",
            "projector",
            "multi_modal_projector",
            "mm_projector",
            "visual_projection",
            "vision_projector",
            "image_projector",
        ],
    )

    matched = _select_top_level_modules(matched)
    return matched


def _log_possible_multimodal_modules(model):
    """
    Debug helper.

    If projector / image encoder cannot be found, this prints possible
    multimodal-related module names.
    """
    base_model = _unwrap_model(model)

    _log("Possible multimodal-related modules:")

    found = False

    for name, module in base_model.named_modules():
        lname = name.lower()

        if any(
            k in lname
            for k in [
                "vision",
                "visual",
                "image",
                "projector",
                "embed",
                "multi_modal",
                "mm_",
                "siglip",
                "clip",
            ]
        ):
            found = True
            _log(f"  {name}: {module.__class__.__name__}")

    if not found:
        _log("  No obvious multimodal module names found.")


def _pad_sequence(sequences: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    """右側 padding，回傳 [batch, max_len] tensor。"""
    max_len = max(s.size(0) for s in sequences)
    batch = sequences[0].new_full((len(sequences), max_len), padding_value)

    for i, seq in enumerate(sequences):
        batch[i, :seq.size(0)] = seq

    return batch
