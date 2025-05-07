# fxt200001
# Fengye Tao
# This is a part of the final project code for the course "Natural Language Processing" CS6320

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from datasets import load_dataset
import torch

# Load the model and tokenizer
model_name = "bigscience/T0_3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")


# Define examples with task names and input prompts
example_inputs = [
    {
        "task": "Natural Language Inference (RTE/ANLI)",
        "prompt": "Premise: The man is walking his dog. Hypothesis: A man is out with his pet. Is the hypothesis true?"
    },
    {
        "task": "Question Answering (BoolQ)",
        "prompt": "Passage: Lightning can strike the same place twice. Question: Can lightning strike the same place twice?"
    },
    {
        "task": "Coreference Resolution (WSC)",
        "prompt": "Sentence: Emma did not pass the ball to Janie although she saw that she was open. Question: Who was open?"
    },
    {
        "task": "Causal Reasoning (COPA)",
        "prompt": "Premise: The man broke his toe. What was the cause? Choice1: He got a hole in his sock. Choice2: He dropped a hammer on his foot."
    },
    {
        "task": "Sentence Completion (HellaSwag)",
        "prompt": "First, I opened the cupboard. Then, I reached for the cereal. Next, I poured it into the bowl. Finally,"
    },
    {
        "task": "Multiple-Choice QA (ARC/OpenBookQA)",
        "prompt": "Question: Which part of the plant makes seeds? Options: A. root B. stem C. flower D. leaf"
    },
    {
        "task": "Word Sense Disambiguation (WiC)",
        "prompt": "Sentence 1: The bank will not be open on Sunday. Sentence 2: He sat down on the river bank. Question: Do the two sentences use the word 'bank' in the same way?"
    },
]

# Run inference on each example
for idx, example in enumerate(example_inputs, 1):
    print(f"\n🔹 Example {idx}: Task - {example['task']}")
    print(f"📝 Prompt: {example['prompt']}")
    
    inputs = tokenizer(example["prompt"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64)
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🔸 Model Output: {decoded_output}")