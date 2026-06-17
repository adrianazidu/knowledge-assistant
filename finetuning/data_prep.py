"""finetuning/data_prep.py — format raw training examples for LLaMA 3."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

def format_example(ex: dict) -> str:
    user = f"{ex['instruction']}\n\n{ex['input']}" if ex.get("input","").strip() else ex["instruction"]
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{config.FINETUNING_SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{ex['output']}<|eot_id|>"
    )

def run(raw_path: str = None, preview: bool = False):
    raw_path = raw_path or "data/raw/training_examples.json"
    print(f"\n  Loading: {raw_path}")
    with open(raw_path) as f:
        examples = json.load(f)

    formatted = [format_example(e) for e in examples]

    if preview:
        print("\n" + "="*60)
        print(formatted[0])
        print("="*60)
        return

    random.seed(42)
    data = list(zip(examples, formatted))
    random.shuffle(data)
    split     = int(len(data) * 0.85)
    train, val = data[:split], data[split:]

    os.makedirs(os.path.dirname(config.FORMATTED_DATA_PATH), exist_ok=True)
    train_path = config.FORMATTED_DATA_PATH.replace(".jsonl","_train.jsonl")
    val_path   = config.FORMATTED_DATA_PATH.replace(".jsonl","_val.jsonl")

    for path, items in [(train_path, train), (val_path, val)]:
        with open(path,"w") as f:
            for _, text in items:
                f.write(json.dumps({"text": text}) + "\n")

    print(f"  Train: {len(train)} | Val: {len(val)}")
    print(f"  Saved: {train_path}")
    return train_path, val_path
