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