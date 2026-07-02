"""finetuning/data_prep.py — format raw training examples for LLaMA 3.

What ChromaDB is for
ChromaDB stores your documents so they can be retrieved at query time. When a user asks a question, the system searches ChromaDB and pulls relevant 
chunks to give the LLM as ***context***. The LLM reads them and answers.
The LLM itself doesn't learn anything from ChromaDB. It just reads what gets retrieved, answers, and forgets it. Next query starts fresh.

What fine-tuning is for
Fine-tuning permanently changes the model's weights — it teaches the model how to behave, not what to know. Tone, reasoning style, output format, refusals.
The training data for fine-tuning comes from training_examples.json — the instruction/input/output pairs you generate with generate.py.
"""
import json, os, random, sys
#allows your script to import modules from two folder levels up (config)
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
    raw_path = raw_path or "data/training_examples.json"
    print(f"\n  Loading: {raw_path}")
    with open(raw_path) as f:
        examples = json.load(f) #load json inside file

    formatted = [format_example(e) for e in examples]

    if preview:
        print("\n" + "="*60)
        print(formatted[0])
        print("="*60)
        return

    random.seed(42)                         #t locks the random number generator to a specific "seed" value (42 is just a popular choice in the programming community).
    data = list(zip(examples, formatted))   #glue examples and formatted together (zip)
    random.shuffle(data)                    #randomly mix up the order
    split     = int(len(data) * 0.85)       #compute index where list is art 85%
    train, val = data[:split], data[split:] #split data using the index into traing and test data (validation)

    os.makedirs(os.path.dirname(config.FORMATTED_DATA_PATH), exist_ok=True)
    train_path = config.FORMATTED_DATA_PATH.replace(".jsonl","_train.jsonl")
    val_path   = config.FORMATTED_DATA_PATH.replace(".jsonl","_val.jsonl")

    #write the two lists to two different files
    for path, items in [(train_path, train), (val_path, val)]:
        with open(path,"w") as f:
            for _, text in items:
                f.write(json.dumps({"text": text}) + "\n")

    print(f"  Train: {len(train)} | Val: {len(val)}")
    print(f"  Saved: {train_path}")
    return train_path, val_path
