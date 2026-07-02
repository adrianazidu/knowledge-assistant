"""finetuning/trainer.py 
trainer.py is the fine-tuning script. It takes your formatted training data (training_examples.json after being processed by data_prep.py) 
and actually trains the model.

It does four things in sequence:
1. Loads the base model — downloads LLaMA or Mistral from HuggingFace and loads it into GPU memory in 4-bit quantization to fit on a single GPU.
2. Attaches LoRA adapters — adds small trainable matrices on top of the frozen base model. Only these adapters get trained, not the full model. 
    This is what makes fine-tuning fit on a single GPU instead of requiring 8x A100s.
3. Trains — runs the training loop using SFTTrainer from the trl library.
     For each example in your dataset it shows the model the instruction and input, asks it to predict the output, measures how wrong it was (loss), 
     and adjusts the LoRA adapter weights slightly to do better next time. Repeats for NUM_EPOCHS passes through the entire dataset.

Fine-tuning using pure HuggingFace — no Unsloth required.
Works on CPU (slow) and GPU (fast).
 
Libraries used:
  transformers → model architecture, tokenizer, loading from HuggingFace hub
  peft         → LoRA adapter implementation
  trl          → SFTTrainer (supervised fine-tuning training loop)
  datasets     → loading and processing training data
  torch        → underlying math engine (used automatically, never called directly)
 
Install:
  pip install transformers peft trl datasets torch accelerate

  Steps
  ----------
load_dataset(train_path)          # 1. load your training examples into memory
        ↓
SFTTrainer(train_dataset=...)     # 2. configure trainer with those examples
        ↓
trainer.train()                   # 3. LoRA weights updated in memory (model object)
        ↓
model.save_pretrained(...)        # 4. LoRA weights written to disk
        ↓
PeftModel.from_pretrained(...)    # 5. reload base + trained adapters
        ↓
merge_and_unload()                # 6. combine into one model
        ↓
merged_model.save_pretrained(...) # 7. save final standalone model
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

 
def run(dry_run: bool = False, resume: bool = False, merge: bool = True):
 
    # ── Check dependencies ────────────────────────────────────────────────────
    # Import everything here instead of at module level.
    # This way the rest of the project works even if these packages
    # are not installed — only fails when you actually try to train.
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,   # loads any causal LM from HuggingFace hub
            AutoTokenizer,          # loads the matching tokenizer
            TrainingArguments,      # hyperparameter configuration object
        )
        from peft import (
            LoraConfig,             # defines LoRA adapter settings
            get_peft_model,         # applies LoRA adapters to a model
            TaskType,               # tells PEFT what kind of task (causal LM)
            PeftModel,              # used when loading saved adapters
        )

        #Transformer Reinforcement Learning
        from trl import SFTTrainer  # supervised fine-tuning trainer
        from datasets import load_dataset  # loads jsonl files into Dataset objects
 
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   Run: pip install transformers peft trl datasets torch accelerate")
        return
 
    # ── Check training data exists ────────────────────────────────────────────
    # data_prep.py must have been run first to produce these files.
    # They contain the formatted instruction/input/output examples
    # in LLaMA/Mistral chat format with special tokens.
    train_path = config.FORMATTED_DATA_PATH.replace(".jsonl", "_train.jsonl")
    val_path   = config.FORMATTED_DATA_PATH.replace(".jsonl", "_val.jsonl")
 
    if not os.path.exists(train_path):
        print(f"❌ No training data at {train_path}")
        print("   Run: python main.py finetune prep")
        return
 
    # ── Detect device ─────────────────────────────────────────────────────────
    # torch.device tells PyTorch where to run computations.
    # "cuda" = NVIDIA GPU (fast)
    # "mps"  = Apple Silicon GPU (medium)
    # "cpu"  = CPU only (slow but works)
    # This auto-detects the best available option.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n  Device: GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n  Device: Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("\n  Device: CPU (training will be slow)")

    #Download starts - hf saves all in users/cache/huggingface so if you run the script multiple times, it won't download it again
    #to change the download path from C to other , put a HF_HOME="your other folder" in .env file

    """
    We cannot use Ollama model for fine-tuning 
    Ollama runs a model as a black box server. You send it text, it sends back text. You have no access to the actual weights, no ability to compute gradients through it,
    no way to update its parameters. Fine-tuning requires direct access to the model weights in memory so PyTorch can run backpropagation through them.
    """
 
    # ── Load tokenizer ────────────────────────────────────────────────────────
    # The tokenizer converts text → token IDs (numbers the model understands)
    # and token IDs → text.
    # Every model has its own vocabulary and special tokens, so the tokenizer
    # must match the model exactly.
    # AutoTokenizer automatically downloads the right one from HuggingFace.
    print(f"\n── Loading tokenizer: {config.active_base_model()}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.active_base_model(),
        trust_remote_code=True,   # some models need custom code from the repo
    )
 
    # LLaMA and Mistral don't have a padding token by default.
    # SFTTrainer needs one to pad shorter examples in a batch to the same length.
    # We use the end-of-sequence token as the padding token — common practice.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  Set pad_token = eos_token")
 
    # ── Load base model ───────────────────────────────────────────────────────
    # AutoModelForCausalLM downloads the model weights from HuggingFace.
    # "Causal LM" means it predicts the next token given all previous tokens —
    # which is exactly how LLaMA and Mistral work.
    #
    # torch_dtype controls numerical precision:
    #   torch.float16  → 16-bit, saves memory, works on most GPUs
    #   torch.bfloat16 → better for training stability on Ampere+ GPUs
    #   torch.float32  → full precision, required for CPU
    #
    # device_map="auto" tells HuggingFace to automatically spread the model
    # across available GPUs (or CPU if no GPU). Without this you'd have to
    # manually move tensors to the right device.
    print(f"  Loading model weights...")
    dtype = torch.float32 if device.type == "cpu" else torch.float16
 
    model = AutoModelForCausalLM.from_pretrained(
        config.active_base_model(),
        dtype=dtype,
        device_map="auto",        # automatically places layers on available devices
        trust_remote_code=True,
    )
 
    # Enable gradient checkpointing — trades compute for memory.
    # Instead of storing all intermediate activations during the forward pass
    # (needed for backpropagation), it recomputes them during the backward pass.
    # Uses ~30% more compute but cuts memory usage significantly.
    # Critical for CPU training where RAM is limited.
    model.gradient_checkpointing_enable()
 
    # ── Apply LoRA adapters ───────────────────────────────────────────────────
    # LoraConfig defines which layers to add adapters to and how large they are.
    #
    # r (rank): size of the adapter matrices. Higher = more capacity to learn
    #   but more memory and compute. 8-32 is typical. We use config.LORA_RANK.
    #
    # lora_alpha: scaling factor. Usually set equal to r.
    #   The actual scaling applied is lora_alpha/r, so equal values → scale of 1.
    #
    # target_modules: which attention layers to add adapters to.
    #   q_proj = query projection, v_proj = value projection.
    #   These are the most impactful layers to fine-tune in transformer models.
    #   You could also add k_proj, o_proj, gate_proj etc. for more capacity.
    #
    # lora_dropout: randomly zeros some adapter weights during training.
    #   Prevents overfitting on small datasets. 0.05 = 5% of weights zeroed per step.
    #
    # bias="none": don't train bias terms, only the adapter matrices.
    #   Standard practice — biases add little quality but cost memory.
    #
    # task_type=CAUSAL_LM: tells PEFT this is a text generation task,
    #   so it sets up the adapter correctly for autoregressive generation.
    print("  Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],   # attention layers to adapt
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
 
    # get_peft_model wraps the base model with LoRA adapters.
    # It freezes all original weights (requires_grad=False) and only
    # the new adapter matrices remain trainable (requires_grad=True).
    model = get_peft_model(model, lora_config)
 
    # Count and display how many parameters will actually be updated.
    # For an 8B model with r=16 this is typically ~20M out of 8B — about 0.25%.
    # This is why LoRA fits on a single GPU.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} ({100*trainable/total:.3f}% of {total:,})")
    print(f"  Frozen parameters:    {total-trainable:,} (base model — not updated)")
 
    # ── Load dataset ──────────────────────────────────────────────────────────
    # TRAIN MODEL WITH OUR FILES
    # load_dataset reads the jsonl files produced by data_prep.py.
    # Each line is a JSON object with a "text" field containing the full
    # formatted training example (system + user + assistant in chat format).
    # HuggingFace Dataset handles batching, shuffling, and tokenization.
    print(f"\n  Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train":      train_path,
            "validation": val_path,
        }
    )
    print(f"  Train examples:      {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['validation'])}")
 
    # Stop here if dry run — everything loaded successfully
    if dry_run:
        print("\n  ✅ Dry run complete — model and data loaded OK. Ready to train.")
        return
 
    # ── Configure training ────────────────────────────────────────────────────
    # TrainingArguments holds all hyperparameters.
    # These control HOW training happens, not what the model learns.
    os.makedirs(config.FINAL_MODEL_DIR, exist_ok=True)
 
    training_args = TrainingArguments(
        output_dir=config.FINAL_MODEL_DIR,
 
        # How many times to loop through the entire training dataset.
        # More epochs = more learning but risk of overfitting.
        num_train_epochs=config.NUM_EPOCHS,
 
        # How many examples to process at once per device.
        # Larger = faster but uses more memory. Reduce if you get OOM errors.
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
 
        # Gradient accumulation: process N small batches but only update
        # weights after all N. Simulates a larger effective batch size
        # (effective_batch = BATCH_SIZE * GRAD_ACCUM) without extra memory.
        gradient_accumulation_steps=config.GRAD_ACCUM,
 
        # Learning rate: how much to adjust weights per step.
        # Too high = unstable training. Too low = learns too slowly.
        # 2e-4 is standard for LoRA fine-tuning.
        learning_rate=config.LEARNING_RATE,
 
        # Warmup: gradually increase learning rate from 0 to learning_rate
        # over the first N steps. Prevents large destabilizing updates at start.
        warmup_steps=5,
 
        # Weight decay: small penalty on large weights. Prevents overfitting.
        weight_decay=0.01,
 
        # Log training loss every N steps so you can watch progress.
        logging_steps=1,
 
        # Save a checkpoint after every epoch so you can resume if it crashes.
        save_strategy="epoch",
 
        # Run validation after every epoch to measure quality on unseen data.
        eval_strategy="epoch",
 
        # At the end, load whichever checkpoint had the best validation loss
        # rather than the final one (which might have overfit).
        load_best_model_at_end=True,
 
        # Optimizer: AdamW is standard for transformer fine-tuning.
        # "adamw_torch" is the CPU-compatible version.
        # On GPU you could use "adamw_8bit" (from bitsandbytes) to save memory.
        optim="adamw_torch",
 
        # Numerical precision for training.
        # fp16=True on GPU saves memory. Must be False on CPU.
        # bf16 is better for Ampere+ GPUs but not supported everywhere.
        fp16=False,
        bf16=False,   # set True if you have an Ampere GPU (RTX 3000+, A100)
 
        # Don't send metrics to Weights & Biases or other trackers.
        report_to="none",
 
        # Random seed for reproducibility — same seed = same results.
        seed=42,
    )
 
    # ── Create trainer ────────────────────────────────────────────────────────
    #PASS THE DATASET CREATED EARLIER AS FINETUNING DATA
    # SFTTrainer (Supervised Fine-Tuning Trainer) wraps TrainingArguments
    # and handles the training loop automatically.
    #
    # Key thing it does that plain Trainer doesn't:
    # It applies a labels mask so the model only learns to predict the
    # ASSISTANT part of each example. The system prompt and user message
    # are masked with -100 so they don't contribute to the loss.
    # Without this the model would try to learn to predict the user's
    # question too, which makes no sense.
    #
    # dataset_text_field="text": tells SFTTrainer which field in your
    # jsonl to use. Our data_prep.py puts the full formatted text in "text".
    #
    # max_seq_length: truncates examples longer than this many tokens.
    # Longer = more context but more memory.
    print("\n── Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        args=training_args,
    )
 
    # ── Train ─────────────────────────────────────────────────────────────────
    # This runs the actual training loop:
    # for each epoch:
    #   for each batch:
    #     1. forward pass  — model reads input, predicts output tokens
    #     2. compute loss  — how wrong was the prediction? (cross-entropy)
    #     3. backward pass — calculate gradients (how to adjust each weight)
    #     4. update weights — adjust LoRA adapter weights by learning_rate * gradient
    #   evaluate on validation set
    #   save checkpoint
    print("\n── Training...")
    print(f"  Epochs:         {config.NUM_EPOCHS}")
    print(f"  Batch size:     {config.BATCH_SIZE}")
    print(f"  Grad accum:     {config.GRAD_ACCUM} (effective batch: {config.BATCH_SIZE * config.GRAD_ACCUM})")
    print(f"  Learning rate:  {config.LEARNING_RATE}")
    print(f"  Checkpoints  →  {config.FINAL_MODEL_DIR}\n")
 
    #DO THE ACTUAL TRAINING
    trainer.train(
        # If --resume was passed, continue from the last saved checkpoint
        # instead of starting from scratch. Useful if training crashed.
        resume_from_checkpoint=config.FINAL_MODEL_DIR if resume else None
    )
 
    # ── Save LoRA adapters ────────────────────────────────────────────────────
    # save_pretrained saves only the LoRA adapter weights — NOT the base model.
    # The adapter file is small (~50MB for rank 16) because it only contains
    # the delta weights, not the full 8B parameter model.
    #
    # To use these adapters later you need both:
    #   1. The original base model (downloaded from HuggingFace)
    #   2. These adapter weights (saved here)
    # PeftModel.from_pretrained(base_model, adapter_path) loads both together.
    print("\n── Saving...")
    model.save_pretrained(config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(config.FINAL_MODEL_DIR)
    print(f"  ✅ LoRA adapters saved → {config.FINAL_MODEL_DIR}")
    print(f"     (small file — needs base model to run)")
 
    # ── Merge adapters into base model (optional) ─────────────────────────────
    # Merging bakes the LoRA weights permanently into the base model weights.
    # Result: a single standalone model file with no adapter dependency.
    #
    # Before merge: base_model + adapter = your fine-tuned model (two files)
    # After merge:  merged_model         = your fine-tuned model (one file, ~16GB)
    #
    # Use the merged version when:
    #   - Serving with vLLM (simpler setup)
    #   - Sharing the model as a standalone artifact
    #   - Converting to GGUF for llama.cpp/Ollama
    #
    # Use the adapter version when:
    #   - You want to swap multiple adapters on one base model
    #   - You want to keep the file size small
    if merge:
        print("\n── Merging adapters into base model...")
        print("  (This creates a standalone model — no adapter dependency)")
 
        # Load the base model again in full precision for merging.
        # We can't merge into a quantized model, so we reload in float16.
        from peft import PeftModel
 
        print("  Loading base model for merge...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.active_base_model(),
            dtype=torch.float16,
            device_map="auto",
        )
 
        # Load the LoRA adapters on top of the base model
        print("  Applying adapters...")
        merged_model = PeftModel.from_pretrained(base_model, config.FINAL_MODEL_DIR)
 
        # merge_and_unload() mathematically combines adapter weights into
        # the base model weights and removes the adapter structure.
        # Result is a plain model with no PEFT overhead.
        print("  Merging weights...")
        merged_model = merged_model.merge_and_unload()
 
        # Save the merged model
        os.makedirs(config.MERGED_MODEL_DIR, exist_ok=True)
        merged_model.save_pretrained(config.MERGED_MODEL_DIR)
        tokenizer.save_pretrained(config.MERGED_MODEL_DIR)
        print(f"  ✅ Merged model saved → {config.MERGED_MODEL_DIR}")
        print(f"     (standalone — no base model needed, ~16GB)")


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
