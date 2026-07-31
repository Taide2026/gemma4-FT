from config.common import COMMON_TRAINING_DEFAULTS


TRAINING_PROFILE = {
    **COMMON_TRAINING_DEFAULTS,
    "data_path": "./dataset/gemma-4-e4b-kinetics_54K/annotations/splits/train.json",
    "eval_data_path": "./dataset/gemma-4-e4b-kinetics_54K/annotations/splits/val.json",
    "image_folder": "./dataset/gemma-4-e4b-kinetics_54K",
    "output_dir": "./output/gemma4_e4b_action_stage1",
    "run_name": "gemma-4-e4b-kinetics54K_LoRA",
    "training_mode": "lora",
    "num_gpus": 4,
    "optim": "adamw_torch",
    "learning_rate": 1e-5,
    "image_encoder_lr": 1e-5,
    "projector_lr": 1e-5,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "report_to": "wandb",
}
