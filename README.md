run the program in python 3.9 environment

1. recommend to use a virtual environment
```bash
myenv\Scripts\activate
```
to activate the virtual environment

2. installed the required package
*promptsource
to install promptsource, please follow the instruction of the link: https://github.com/bigscience-workshop/promptsource, if you use python >3.7, you need also modify the version in setup.py.

*t-zero
t-zero is a utility package that are designed to simplify tasks like data preprocessing, evaluation, or training for specific use cases.
to install t-zero, please follow the instruction of the link: https://github.com/bigscience-workshop/t-zero, if you use python >3.7, you need also modify the version in setup.py.

*torch
enable gpu and cuda is essential for this project
for example, install Pytorch with CUDA 11.8, use:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

other needed package in the requirements.txt

3. reproduce the evaluation -- run_eval.py

example usage:
```bash
python run_eval.py \
    --dataset_name super_glue \
    --dataset_config_name rte \
    --template_name "must be true" \
    --model_name_or_path bigscience/T0_3B \
    --output_dir ./debug
```

4. reproduce the inference -- inference.py
```bash
python inference.py
```
