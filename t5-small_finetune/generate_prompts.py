# generate_prompts.py
from datasets import load_dataset
from promptsource.templates import DatasetTemplates
import random
import json
import os

DATASETS = [
    ("super_glue", "copa"),
    ("super_glue", "wsc.fixed"),
    ("winogrande", "winogrande_xl")
]

def apply_all_templates(example, dataset_name, config_name):
    templates = DatasetTemplates(dataset_name, config_name)
    results = []
    for template in templates.templates.values():
        try:
            input_text, target_text = template.apply(example)
            if input_text and target_text:
                results.append({"input_text": input_text, "target_text": target_text})
        except:
            continue
    return results

def apply_random_template(example, dataset_name, config_name):
    templates = DatasetTemplates(dataset_name, config_name)
    template = random.choice(list(templates.templates.values()))
    try:
        input_text, target_text = template.apply(example)
        return {"input_text": input_text, "target_text": target_text}
    except:
        return None

os.makedirs("prompted_data", exist_ok=True)

for dataset_name, config_name in DATASETS:
    print(f"Processing {dataset_name}/{config_name}")
    
    raw_train = load_dataset(dataset_name, config_name, split="train")
    raw_val = load_dataset(dataset_name, config_name, split="validation")

    train_prompted = []
    for ex in raw_train:
        prompted = apply_all_templates(ex, dataset_name, config_name)
        train_prompted.extend(prompted)

    val_prompted = []
    for ex in raw_val:
        prompted = apply_random_template(ex, dataset_name, config_name)
        if prompted:
            val_prompted.append(prompted)

    # Save to disk
    train_path = f"prompted_data/{dataset_name}_{config_name}_train.json"
    val_path = f"prompted_data/{dataset_name}_{config_name}_val.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_prompted, f, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_prompted, f, ensure_ascii=False)

    print(f"Saved: {train_path} and {val_path}")
