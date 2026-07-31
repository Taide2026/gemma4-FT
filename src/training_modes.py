from peft import LoraConfig, get_peft_model
from arguments import GemmaSFTTrainingArguments, ModelArguments
from utils import (
    _freeze_all,
    _freeze_llm,
    _freeze_image_encoder,
    _log,
    _unfreeze_image_encoder,
    _unfreeze_projector,
    _print_trainable_parameters,
)


def apply_training_mode(
    model,
    model_args: ModelArguments,
    training_args: GemmaSFTTrainingArguments,
    compute_dtype,
):
    device = training_args.device

    image_encoder_lr = getattr(training_args, "image_encoder_lr", 0.0)
    projector_lr = getattr(training_args, "projector_lr", 0.0)

    if model_args.training_mode == "full":
        for param in model.parameters():
            param.requires_grad = True

        _log("Training mode: full fine-tuning")
        _print_trainable_parameters(model)
        return model

    if model_args.training_mode == "projector_only":
        # Safer behavior:
        # freeze everything first, then explicitly unfreeze projector / image encoder.
        _freeze_all(model)

        if projector_lr > 0:
            _unfreeze_projector(model, compute_dtype, device)
            _log(f"Projector-only mode: projector is unfrozen, projector_lr={projector_lr}")
        else:
            _log("Projector-only mode: projector remains frozen because projector_lr <= 0")

        if image_encoder_lr > 0:
            _unfreeze_image_encoder(model, compute_dtype, device)
            _log(
                f"Projector-only mode: image encoder is unfrozen, "
                f"image_encoder_lr={image_encoder_lr}"
            )
        else:
            _freeze_image_encoder(model)
            _log("Projector-only mode: image encoder remains frozen")

        _log("Training mode: projector-only fine-tuning")
        _print_trainable_parameters(model)
        return model

    if model_args.training_mode == "lora":
        # Freeze language model first.
        # PEFT will make LoRA adapter weights trainable after get_peft_model().
        _freeze_llm(model)

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            layers_to_transform=list(range(42)),  # gemma4-e4b has 42 text layers
            layers_pattern="language_model.layers",
        )

        model = get_peft_model(model, lora_config)

        # Important:
        # Do this AFTER get_peft_model().
        # PEFT may freeze base model parameters, so projector / image encoder
        # must be explicitly unfrozen after wrapping.
        if projector_lr > 0:
            _unfreeze_projector(model, compute_dtype, device)
            _log(f"LoRA mode: projector is unfrozen, projector_lr={projector_lr}")
        else:
            _log("LoRA mode: projector remains frozen because projector_lr <= 0")

        if image_encoder_lr > 0:
            _unfreeze_image_encoder(model, compute_dtype, device)
            _log(f"LoRA mode: image encoder is unfrozen, image_encoder_lr={image_encoder_lr}")
        else:
            _freeze_image_encoder(model)
            _log("LoRA mode: image encoder remains frozen")

        _log("Training mode: LoRA fine-tuning")
        _print_trainable_parameters(model)
        return model

    raise ValueError(f"Unsupported training_mode: {model_args.training_mode}")


def configure_gradient_checkpointing(
    model,
    training_args: GemmaSFTTrainingArguments,
):
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
