"""finetuning/trainer.py — LoRA fine-tuning with Unsloth."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

def run(dry_run: bool = False, resume: bool = False, merge: bool = True):
    try:
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("❌ Missing packages. Run: pip install unsloth trl transformers datasets")
        return

    train_path = config.FORMATTED_DATA_PATH.replace(".jsonl","_train.jsonl")
    val_path   = config.FORMATTED_DATA_PATH.replace(".jsonl","_val.jsonl")

    if not os.path.exists(train_path):
        print(f"❌ No training data at {train_path}. Run: python main.py finetune prep")
        return

    print(f"\n── Loading model: {config.BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL, max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=config.LOAD_IN_4BIT, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=config.LORA_RANK, target_modules="all-linear",
        lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
        bias="none", use_gradient_checkpointing="unsloth", random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    dataset = load_dataset("json", data_files={"train":train_path,"validation":val_path})
    print(f"  Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

    if dry_run:
        print("  ✅ Dry run — model and data loaded OK. Ready to train.")
        return

    os.makedirs(config.FINAL_MODEL_DIR, exist_ok=True)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        dataset_text_field="text", max_seq_length=config.MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=config.FINAL_MODEL_DIR,
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUM,
            learning_rate=config.LEARNING_RATE,
            warmup_steps=5, weight_decay=0.01,
            logging_steps=1, save_strategy="epoch", evaluation_strategy="epoch",
            load_best_model_at_end=True, optim="adamw_8bit",
            bf16=True, fp16=False, report_to="none", seed=42,
        ),
    )
    trainer.train(resume_from_checkpoint=config.FINAL_MODEL_DIR if resume else None)

    model.save_pretrained(config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(config.FINAL_MODEL_DIR)
    print(f"  ✅ LoRA adapters → {config.FINAL_MODEL_DIR}")

    if merge:
        os.makedirs(config.MERGED_MODEL_DIR, exist_ok=True)
        model.save_pretrained_merged(config.MERGED_MODEL_DIR, tokenizer, save_method="merged_16bit")
        print(f"  ✅ Merged model → {config.MERGED_MODEL_DIR}")

def serve(mode: str = "merged", port: int = 8000):
    import subprocess
    if mode == "merged":
        path = config.MERGED_MODEL_DIR
        cmd  = ["vllm","serve", path,"--host","0.0.0.0","--port",str(port)]
    else:
        lora = f"finetuned={config.FINAL_MODEL_DIR}"
        cmd  = ["vllm","serve", config.BASE_MODEL,"--enable-lora",
                "--lora-modules", lora,"--host","0.0.0.0","--port",str(port)]
    print(f"\n  Starting vLLM: {' '.join(cmd)}\n")
    subprocess.run(cmd)
