###########################################
# Fengye Tao 
# fxt200001
# CS6320 Natural Language Processing
###########################################
# this script is the original code for finetuning T5 on multiple datasets
# But, as it may burn my laptop, if I run it locally. I decided to run it on some cloud platform with advanced GPUs
# The problem I encountered is the "promptsource" requires a very old "click" version (= 7.1.2). I failed to run this Script on colab, kaggle, and gradient paperspace.
# finally, I come up with a solution to seperate the code into two parts: code for generating prompt templates "generate_prompts.py" and code for train the model "train.py".
# I really enjoy this project as it provides me good examples of BQ in interview.

from datasets import load_dataset, concatenate_datasets
from promptsource.templates import DatasetTemplates
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
import random
import time


# Check CUDA availability
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

# Set model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# List of datasets and configs
DATASETS = [
    # ("super_glue", "boolq")
    # ("super_glue", "wic"),
    # t5-small is already trained on these tasks
    ("super_glue", "copa"),
    ("super_glue", "wsc.fixed"),
    ("winogrande", "winogrande_xl")
]

# Function to apply all templates for training
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

# Function to apply a random template for validation
def apply_random_template(example, dataset_name, config_name):
    templates = DatasetTemplates(dataset_name, config_name)
    template = random.choice(list(templates.templates.values()))
    try:
        input_text, target_text = template.apply(example)
        return {"input_text": input_text, "target_text": target_text}
    except:
        return None

# Tokenize function
def preprocess(example):
    inputs = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(example["target_text"], truncation=True, padding="max_length", max_length=32)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Load, apply templates, preprocess, and combine datasets
train_datasets, val_datasets = [], []

for dataset_name, config_name in DATASETS:
    print(f"Loading {dataset_name}/{config_name}")

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

    train_tokenized = list(map(preprocess, train_prompted))
    val_tokenized = list(map(preprocess, val_prompted))

    train_datasets.append(train_tokenized)
    val_datasets.append(val_tokenized)

# Convert lists to datasets
from datasets import Dataset
full_train = concatenate_datasets([Dataset.from_list(d) for d in train_datasets])
full_val = concatenate_datasets([Dataset.from_list(d) for d in val_datasets])

# Set format for PyTorch
full_train.set_format(type="torch")
full_val.set_format(type="torch")

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5-small-multitask",
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    save_total_limit=2
)

# Trainer setup
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_train,
    eval_dataset=full_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
print("\n===== Starting training... =====")
start_time = time.time()
trainer.train()
end_time = time.time()
print(f"\n===== Training completed in {round((end_time - start_time)/60, 2)} minutes =====")

# Evaluate
print("\n===== Starting evaluation... =====")
metrics = trainer.evaluate()
print("\nEvaluation metrics:", metrics)


from sklearn.metrics import accuracy_score

# Generate predictions
print("\n===== Generating predictions for accuracy evaluation... =====")
predictions = trainer.predict(full_val)

# Decode predictions and labels
decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

# Normalize (strip and lowercase)
decoded_preds = [pred.strip().lower() for pred in decoded_preds]
decoded_labels = [label.strip().lower() for label in decoded_labels]

# Compute accuracy
acc = accuracy_score(decoded_labels, decoded_preds)
print(f"\nValidation Accuracy: {acc:.4f}")