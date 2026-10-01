COMMON_TRAINING_DEFAULTS = {
    "model_name": "THChou1220/gemma-4-e4b-kinetics54K-MQ_FFT",
    "deepspeed_config": "deepspeed_config/stage1.json",
    "master_port": "29500",
    "bf16": "False",
    "fp16": "True",
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "gradient_checkpointing": "True",
    "logging_steps": 10,
    "dataloader_num_workers": 4,
    "max_seq_length": 3072,
    "learning_rate": 5e-6,
    "image_encoder_lr": 5e-6,
    "projector_lr": 5e-6,
    "ple_enabled": "True",
    "ple_trainable": "True",
    "seed": 42,
    "load_best_model_at_end": "False",
    "metric_for_best_model": "eval_loss",
    "weight_decay": 0.01,
    "num_train_epochs": 1,
    "save_total_limit": 1,
}


SHELL_ENV_NAMES = {
    "model_name": "MODEL_NAME",
    "model_revision": "MODEL_REVISION",
    "data_path": "DATA_PATH",
    "eval_data_path": "EVAL_DATA_PATH",
    "image_folder": "IMAGE_FOLDER",
    "output_dir": "OUTPUT_DIR",
    "run_name": "RUN_NAME",
    "deepspeed_config": "DEEPSPEED_CONFIG",
    "num_gpus": "NUM_GPUS",
    "master_port": "MASTER_PORT",
    "bf16": "BF16",
    "fp16": "FP16",
    "num_train_epochs": "NUM_TRAIN_EPOCHS",
    "per_device_train_batch_size": "PER_DEVICE_TRAIN_BATCH_SIZE",
    "per_device_eval_batch_size": "PER_DEVICE_EVAL_BATCH_SIZE",
    "gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
    "optim": "OPTIM",
    "learning_rate": "LEARNING_RATE",
    "image_encoder_lr": "IMAGE_ENCODER_LR",
    "projector_lr": "PROJECTOR_LR",
    "weight_decay": "WEIGHT_DECAY",
    "warmup_ratio": "WARMUP_RATIO",
    "lr_scheduler_type": "LR_SCHEDULER_TYPE",
    "eval_strategy": "EVAL_STRATEGY",
    "load_best_model_at_end": "LOAD_BEST_MODEL_AT_END",
    "metric_for_best_model": "METRIC_FOR_BEST_MODEL",
    "eval_steps": "EVAL_STEPS",
    "save_strategy": "SAVE_STRATEGY",
    "save_steps": "SAVE_STEPS",
    "save_total_limit": "SAVE_TOTAL_LIMIT",
    "gradient_checkpointing": "GRADIENT_CHECKPOINTING",
    "logging_steps": "LOGGING_STEPS",
    "dataloader_num_workers": "DATALOADER_NUM_WORKERS",
    "report_to": "REPORT_TO",
    "max_seq_length": "MAX_SEQ_LENGTH",
    "ple_enabled": "PLE_ENABLED",
    "ple_trainable": "PLE_TRAINABLE",
    "seed": "SEED",
}


def to_shell_defaults(training_profile: dict) -> str:
    lines = []
    for key, env_name in SHELL_ENV_NAMES.items():
        if key not in training_profile:
            continue
        value = str(training_profile[key]).replace('"', '\\"')
        lines.append(f': "${{{env_name}:={value}}}"')
    return "\n".join(lines)
