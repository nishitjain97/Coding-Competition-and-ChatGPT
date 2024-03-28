### Dataset: For dataset_download.py
#
# dataset_names (list): HuggingFace name of dataset to be used. Supports 'deepmind/code_contests' and 'codeparrot/apps'
# dataset_root (str): Root directory for datasets (default: ./dataset)
# verbose (bool): Verbosity (default: False)
# batch_size (int): Size for writing to local file (default: 32)
# n_samples (int): Number of data samples from each dataset
###
dataset_config = {
    'dataset_names': ['deepmind/code_contests',
                       'codeparrot/apps'],
    'dataset_root': './dataset',
    'verbose': True,
    'batch_size': 9,
    'n_samples': 20
}

### Code Generation: For code_generation.py
#
# model_name (str): Name of model to be used. Options include "gpt-3.5-turbo", "gpt-4", "codellama/CodeLlama-7b-Instruct-hf", "codellama/CodeLlama-13b-Instruct-hf" (default: "gpt-3.5-turbo")
# output_root (str): Root directory for output (default: '.')
# dataset_names (list): HuggingFace name of dataset to be used. Supports 'deepmind/code_contests' and 'codeparrot/apps'
# dataset_root (str): Root directory for datasets (default: ./dataset)
# verbose (bool): Verbosity (default: False)
# batch_size (int): Size for writing to local file (default: 32)
# n_samples (int): Number of data samples from each dataset (default: 10)
# k (int): Number of generations (default: 1)
# api_key_location (str): Location of API key for GPT models (default: ./api.txt)
# template_location (str): Location of problem template txt (default: '.')
###
code_gen_config = {
    'model_name': "gpt-3.5-turbo",
    'output_root': '.',
    'dataset_names': ['codeparrot/apps'],
    'dataset_root': './dataset',
    'verbose': False,
    'batch_size': 2,
    'n_samples': 4,
    'k': 1,
    'api_key_location': '/Users/nishitjain/Documents/Nishit/Masters/AP/gpt_api_key.txt',
    'template_location': '.'
}

### Signal Generation: For error_signal.py
#
###
signal_gen_config = {
    'model_name': "gpt-3.5-turbo",
    'output_root': '.',
    'dataset_names': ['codeparrot/apps'],
    'dataset_root': './dataset',
    'verbose': False
}