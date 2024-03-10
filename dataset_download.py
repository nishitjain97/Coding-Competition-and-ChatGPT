# External Libraries
from datasets import load_dataset
import pandas as pd
import os
import json

# Configurations defined in config.py
print("Gathering configs...")
import config as cfg
datasets = cfg.dataset.get('dataset_names', [])
dataset_root = cfg.dataset.get('dataset_root', './dataset')
verbosity = cfg.dataset.get('verbose', False)
print("Configs gathered.")
batch_size = cfg.dataset.get('batch_size', 32)
n_samples = cfg.dataset.get('n_samples', 100)

# Utility Functions
def download_code_contests(
        location: 'str',
        verbosity: bool = False
        ) -> bool:
    """
        Function to download code contests dataset from HuggingFace.
    """
    if not os.path.exists(dataset_location):
        if verbosity:
            print("Creating directory: ", dataset_location)
        
        os.makedirs(dataset_location)

    def download(
            dataset: 'datasets.iterable_dataset.IterableDataset',
            file: 'str'
    ) -> None:
        file_path = os.path.join(location, file)

        if os.path.exists(file_path):
            if verbosity:
                print("File", file_path, "already exists. Deleted existing file.")
            
            os.remove(file_path)

        output_data = {
            'input': [],
            'output': [],
            'tests': []
        }

        if verbosity:
            print("Writing to", file)
            print("Samples: ", end="")

        for i, data in enumerate(dataset):
            if verbosity:
                print(i + 1, end=" ")

            output_data['input'].append(data['description'])
            output_data['output'].append(data['solutions'])
            output_data['tests'].append(json.dumps({
                'public_tests': data['public_tests'],
                'private_tests': {
                    'input': data['private_tests']['input'] + data['generated_tests']['input'],
                    'output': data['private_tests']['output'] + data['generated_tests']['output']
                }
            }))

            if (i + 1) % batch_size == 0:
                pd.DataFrame(output_data).to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

                output_data = {
                    'input': [],
                    'output': [],
                    'tests': []
                }

            if (i + 1) == n_samples:
                break

        pd.DataFrame(output_data).to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

    for split in ['train', 'valid', 'test']:
        dataset = load_dataset('deepmind/code_contests', streaming=True, split=split)
        file = split + '.csv'
        download(dataset, file)

def download_apps(
        location: 'str',
        verbosity: bool = False
        ) -> bool:
    """
        Function to download APPS dataset from HuggingFace.
    """
    if not os.path.exists(dataset_location):
        if verbosity:
            print("Creating directory: ", dataset_location)
        
        os.makedirs(dataset_location)

    def download(
            dataset: 'datasets.iterable_dataset.IterableDataset',
            file: 'str'
    ) -> None:
        file_path = os.path.join(location, file)

        if os.path.exists(file_path):
            if verbosity:
                print("File", file_path, "already exists. Deleted existing file.")
            
            os.remove(file_path)

        output_data = {
            'input': [],
            'output': [],
            'tests': []
        }

        if verbosity:
            print("Writing to", file)
            print("Samples: ", end="")

        for i, data in enumerate(dataset):
            if verbosity:
                print(i + 1, end=" ")

            output_data['input'].append(data['question'])
            output_data['output'].append({
                'language': [3] * len(json.loads(data['solutions'])),
                'solution': json.loads(data['solutions'])
            })
            tests = json.loads(data['input_output'])
            output_data['tests'].append(json.dumps({
                'public_tests': {
                    'input': tests['inputs'],
                    'output': tests['outputs']
                }
            }))

            if (i + 1) % batch_size == 0:
                pd.DataFrame(output_data).to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

                output_data = {
                    'input': [],
                    'output': [],
                    'tests': []
                }

            if (i + 1) == n_samples:
                break

        pd.DataFrame(output_data).to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

    for split in ['train', 'test']:
        dataset = load_dataset('codeparrot/apps', streaming=True, split=split, trust_remote_code=True)
        file = split + '.csv'
        download(dataset, file)

if __name__ == '__main__':
    for dataset in datasets:
        if verbosity:
            print("Downloading: ", dataset)

        dataset_location = os.path.join(dataset_root, dataset)

        if dataset == 'deepmind/code_contests':
            download_code_contests(
                location=dataset_location,
                verbosity=verbosity
            )
        elif dataset == 'codeparrot/apps':
            download_apps(
                location=dataset_location,
                verbosity=verbosity
            )