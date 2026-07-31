from config.common import COMMON_TRAINING_DEFAULTS


TRAINING_PROFILE = {
    **COMMON_TRAINING_DEFAULTS,
    "data_path": "./dataset/gemma-4-e4b-kinetics_54K/annotations/splits-MQ/train.json",
    "eval_data_path": "./dataset/gemma-4-e4b-kinetics_54K/annotations/splits-MQ/val.json",
    "image_folder": "./dataset/gemma-4-e4b-kinetics_54K",
    "output_dir": "./output/gemma4_e4b_action_stage1",
    "run_name": "gemma-4-e4b-kinetics54K-MQ_FFT",
    "training_mode": "full",
    "num_gpus": 4,
    "optim": "adamw_torch",
    "learning_rate": 5e-6,
    "image_encoder_lr": 5e-6,
    "projector_lr": 5e-6,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "report_to": "none",
}
