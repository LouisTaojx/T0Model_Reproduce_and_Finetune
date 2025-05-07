Step 1: Environment Setup

1. It is recommended to use a virtual environment.

2. installed the required package
*promptsource
promptsource provides access to community-curated prompt templates.
Install it by following the instructions on the official GitHub repository:https://github.com/bigscience-workshop/promptsource.

⚠️ Note: If using Python > 3.7, you must manually update the Python version constraint in setup.py.

*t-zero
t-zero is a utility package that simplifies tasks like data preprocessing, evaluation, and training for zero-shot models.
Installation instructions can be found on its GitHub page:https://github.com/bigscience-workshop/t-zero.

⚠️ Note: Similar to promptsource, Python > 3.7 users need to modify the version requirement in setup.py.

*Other Dependencies
Additional required packages are listed in requirements.txt. Install them using:
pip install -r requirements.txt



Step 2: My Work 

Part 1: Reproducing Inference and Evaluation with T0 Models
The goal of Part 1 is to reproduce the inference and evaluation results using T0 models, which are fine-tuned models from the paper "Multitask Prompted Training Enables Zero-Shot Task Generalization".

3. Reproduce the Evaluation — Folder: T0_3B_evaluation
Environment: Run locally with an Nvidia RTX 4050 GPU. Evaluating two datasets takes approximately 3 hours.

Setup:

Clone and install the promptsource library.

⚠️ Python 3.9 is recommended. If using a version higher than 3.7, modify the setup.py file to match your environment.

Install the t-zero library and navigate to t-zero/evaluation/run_eval.py to run the evaluation script.
Example usage:
```bash
python run_eval.py \
    --dataset_name super_glue \
    --dataset_config_name rte \
    --template_name "must be true" \
    --model_name_or_path bigscience/T0_3B \
    --output_dir ./debug
```
Outputs: Example results are included in the subfolder and summarized in the accompanying .docx file.

Note: The evaluation script used in this project is heavily adapted from the original run_eval.py in the t-zero repository.
    
4. Reproduce the Inference — Folder: T0_3B_inference
Environment: Run locally with an Nvidia RTX 4050 GPU. Execution time: 5–10 minutes.

Setup:

This is the easiest part to run. Simply install the required packages: transformers and torch.

Testing: Sample prompt inputs are provided in the script. Feel free to modify or test with your own prompts.

Example usage:
```bash
python inference.py

```


Part 2: Reproducing Fine-Tuning with T5-Small
The goal of Part 2 is to reproduce the fine-tuning procedure described in the T0 paper using multitask learning with multiple prompt templates.

5. Reproduce the Fine-Tuning — Folder: t-small_finetune
Example Attempts:

Attempt I: https://colab.research.google.com/drive/1Q2K2KekRB1i92cr8ACfU1-Zhh_iqazI0#scrollTo=GbDqzPgp2HGc

Ran on Google Colab. The session restarted during execution, so accuracy results were not saved.

Attempt II (Successful workaround): https://colab.research.google.com/drive/1A5G05vkQabusexp-FKnlqxSiKiL7S7gr#scrollTo=F18OOTWmomU6

Split the workflow into two parts:

Ran the prompt generation code locally using the promptsource templates. Saved all outputs in the prompted_data folder.

Uploaded the JSON files to Google Colab and completed training using Python 3.11.

This method effectively bypasses the compatibility issue where promptsource cannot be installed in Colab’s Python 3.11 environment.

I also tested alternative platforms (Kaggle, Gradient by Paperspace) but found this hybrid approach to be the most effective.


Memo: Model parameters
t5_small - 60 million,
t0_3B - 3 billion,
t0, t0pp - 11 billion

