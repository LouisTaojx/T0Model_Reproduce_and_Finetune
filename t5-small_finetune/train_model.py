# train_model.py
from datasets import Dataset, concatenate_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sklearn.metrics import accuracy_score
import json
import torch
import time
import os

DATASETS = [
    ("super_glue", "copa"),
    ("super_glue", "wsc.fixed"),
    ("winogrande", "winogrande_xl")
]

# Load tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load JSON files and preprocess
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess(example):
    inputs = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(example["target_text"], truncation=True, padding="max_length", max_length=32)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_datasets, val_datasets = [], []

for dataset_name, config_name in DATASETS:
    train_path = f"prompted_data/{dataset_name}_{config_name}_train.json"
    val_path = f"prompted_data/{dataset_name}_{config_name}_val.json"

    train_prompted = load_json(train_path)
    val_prompted = load_json(val_path)

    train_tokenized = list(map(preprocess, train_prompted))
    val_tokenized = list(map(preprocess, val_prompted))

    train_datasets.append(Dataset.from_list(train_tokenized))
    val_datasets.append(Dataset.from_list(val_tokenized))

full_train = concatenate_datasets(train_datasets)
full_val = concatenate_datasets(val_datasets)

# Set format for PyTorch
full_train.set_format(type="torch")
full_val.set_format(type="torch")

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5-small-multitask",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_total_limit=2
)

# Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_train,
    eval_dataset=full_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
print("\n===== Starting training... =====")
start_time = time.time()
trainer.train()
end_time = time.time()
print(f"\n===== Training completed in {round((end_time - start_time)/60, 2)} minutes =====")

# Evaluate
print("\n===== Starting evaluation... =====")
metrics = trainer.evaluate()
print("\nEvaluation metrics:", metrics)

# Accuracy score
print("\n===== Accuracy Evaluation... =====")
predictions = trainer.predict(full_val)
decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

decoded_preds = [pred.strip().lower() for pred in decoded_preds]
decoded_labels = [label.strip().lower() for label in decoded_labels]

acc = accuracy_score(decoded_labels, decoded_preds)
print(f"\nValidation Accuracy: {acc:.4f}")
