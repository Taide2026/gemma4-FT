import os
import json
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# Custom Modules
from src.denselora import inject_dense_lora, get_trainable_params
from src.utils import format_and_tokenize
import settings

# Custom Trainer to ONLY save the denseLoRa weights
class DenseLoRATrainer(Trainer):
    # Reference here: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.save_model
    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        """Override save_model to ONLY save DenseLoRA trainable weights."""
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter for only parameters that require gradients
        trainable_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
        lora_weights = {k: v for k, v in self.model.state_dict().items() if k in trainable_names}
        torch.save(lora_weights, os.path.join(output_dir, "denselora_adapters.pt"))
        print(f"\nDenseLoRA weights safely saved to {output_dir}")

# Main Training Loop
def main():
    # Soft Creation if Needed
    os.makedirs(settings.DENSE_LORA_OUTPUT_DIR, exist_ok=True)
    
    # Initialize WandB
    print("Initializing Weights & Biases...")
    wandb.init(
        project="DenseLoRA-Training", # Name of the project in the WandB dashboard
        name=settings.WANDB_RUN_NAME, # The run name
        config={
            "learning_rate": settings.LR,
            "architecture": settings.MODEL_ID,
            "epochs": settings.EPOCHS,
            "batch_size": settings.BATCH_SIZE,
            "rank": settings.RANK,
            "alpha": settings.ALPHA
        }
    )

    print("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)
    if tokenizer.pad_token is None:
        # Safely fallback to eos_token only if the model lacks a native pad_token
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(settings.MODEL_ID, torch_dtype=settings.D_TYPE, device_map=settings.DEVICE)
    model.config.use_cache = False  # Save vRam for training

    print("Injecting DenseLoRA...")
    model = inject_dense_lora(model, settings.RANK, settings.ALPHA, settings.DROPOUT)
    
    get_trainable_params(model)

    print("Processing Data...")
    with open(settings.TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    train_set = Dataset.from_list(raw_data)
    train_set = train_set.map(
        lambda x: format_and_tokenize(x, tokenizer), 
        remove_columns=train_set.column_names,
        desc="Tokenizing & Masking"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    
    print("Configuring Trainer...")
    training_args = TrainingArguments(
        output_dir=settings.DENSE_LORA_OUTPUT_DIR,
        per_device_train_batch_size=settings.BATCH_SIZE,
        gradient_accumulation_steps=settings.GRAD_ACCUM_STEPS,
        learning_rate=settings.LR,
        lr_scheduler_type="linear",
        warmup_steps=settings.WARMUP_STEPS,
        num_train_epochs=settings.EPOCHS,
        logging_steps=settings.LOGGING_STEPS,
        save_strategy="epoch",      # Saves at the end of each epoch using custom logic
        bf16=True,                  # Faster training on RTX 5090
        optim="adamw_torch",
        report_to="wandb",
        gradient_checkpointing=True # Keeps VRAM low
    )

    trainer = DenseLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        data_collator=data_collator,
    )

    print("Starting Training...")
    trainer.train()
    
    # Finish the WandB run cleanly
    wandb.finish()
    
    print("Done!")

if __name__ == "__main__":
    main()
