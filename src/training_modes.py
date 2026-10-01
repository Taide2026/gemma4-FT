from arguments import GemmaSFTTrainingArguments
from utils import _log


def apply_full_finetuning(model):
    for param in model.parameters():
        param.requires_grad = True

    _log("Training mode: full fine-tuning")
    return model


def configure_gradient_checkpointing(
    model,
    training_args: GemmaSFTTrainingArguments,
):
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
