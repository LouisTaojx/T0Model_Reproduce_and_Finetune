from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# Load the model and tokenizer
model_name = "bigscience/T0_3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")


# Example input
input_text = "what is the capital of China?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate output
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))