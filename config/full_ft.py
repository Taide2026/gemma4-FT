from config.common import COMMON_TRAINING_DEFAULTS


TRAINING_PROFILE = {
    **COMMON_TRAINING_DEFAULTS,
    "model_name": "google/gemma-4-E4B-it",
    "model_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "data_path": "./dataset/ucf11/annotations/train.json",
    "eval_data_path": "./dataset/ucf11/annotations/val.json",
    "image_folder": "./dataset/ucf11",
    "output_dir": "./output/ucf11_ple_on",
    "run_name": "ucf11-ple-on",
    "num_train_epochs": 3,
    "num_gpus": 4,
    "optim": "adamw_torch",
    "learning_rate": 5e-6,
    "image_encoder_lr": 5e-6,
    "projector_lr": 5e-6,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": "True",
    "metric_for_best_model": "eval_loss",
    "report_to": "none",
}
