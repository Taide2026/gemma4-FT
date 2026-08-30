from config.common import COMMON_TRAINING_DEFAULTS


TRAINING_PROFILE = {
    **COMMON_TRAINING_DEFAULTS,
    "data_path": "/data/ubuntu/datasets/gemma-4-e4b-kinetics_330K/annotations/train.json",
    "eval_data_path": "/data/ubuntu/datasets/gemma-4-e4b-kinetics_330K/annotations/val.json",
    "image_folder": [
        "/data/ubuntu/datasets/gemma-4-e4b-kinetics_330K"
    ],
    "output_dir": "./output/gemma4_e4b_kinetics54K_MQ_FFT",
    "run_name": "gemma-4-e4b-kinetics54K-MQ_FFT",
    "training_mode": "full",
    "num_gpus": 4,
    "optim": "adamw_torch",
    "learning_rate": 5e-6,
    "image_encoder_lr": 5e-6,
    "projector_lr": 5e-6,
    "eval_strategy": "steps",
    "eval_steps": 1000,
    "save_strategy": "steps",
    "save_steps": 500,
    "report_to": "none",
}
