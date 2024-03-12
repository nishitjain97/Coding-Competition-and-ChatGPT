# External Libraries
import pandas as pd
import os
from openai import OpenAI
from utils import *

# Configurations defined in config.py
print("Gathering configs...")
from config import code_gen_config as cfg
model_name = cfg.get('model_name', 'gpt-3.5-turbo')
output_root = cfg.get('output_root', '.')
dataset_names = cfg.get('dataset_names', [])
dataset_root = cfg.get('dataset_root', './dataset')
verbosity = cfg.get('verbose', False)
batch_size = cfg.get('batch_size', 32)
n_samples = cfg.get('n_samples', 10)
k = cfg.get('k', 1)
temperature = 0.01 if k == 1 else 0.7
api_key_location = cfg.get('api_key_location', '')
template_location = cfg.get('template_location', '.')
print("Configs gathered.")

def get_dataset_generator(
    dataset_location: str,
    nrows: int=None
) -> 'generator':
    file_path = os.path.join(dataset_location, 'train.csv')
    dataset = pd.read_csv(
        filepath_or_buffer=file_path,
        nrows=nrows
    )

    for _, row in dataset.iterrows():
        yield row

def generate_gpt_code(
        model_name: str,
        dataset_name: str,
        dataset_generator: 'generator'
) -> None:
    assert api_key_location != '', "Specify GPT API key in config.py"

    with open(api_key_location, 'r') as f:
        client = OpenAI(
            api_key=f.readlines()[0]
        )
        
        if verbosity:
            print("Loaded OpenAI API Key")

    file_location = os.path.join(output_root, 'results_first_iteration')
    file_path = os.path.join(file_location, f"{dataset_name.replace('/', '-')}_{model_name}.csv")

    if os.path.exists(file_path):
        print(file_path, "exists. Replacing.")
        os.remove(file_path)

    os.makedirs(file_location, exist_ok=True)
    
    with open(os.path.join(template_location, 'prompt_template.txt'), 'r') as f:
        prompt_template = ''.join(f.readlines())

    if verbosity:
        print("Read prompt template.")
        print("Sample: ", end="")

    output_data = {
        'index': [],
        'predict': [],
        'actual': [],
        'tests': []
    }

    for i in range(n_samples):
        if verbosity:
            print(i + 1, end=" ")

        sample = next(dataset_generator)

        problem = sample['input']
        prompt = create_prompt(
            description={
                '%{prompt}%': problem
            },
            prompt_template=prompt_template
        )
        
        responses = ask_chatgpt(
            client=client,
            prompt=prompt,
            model=model_name,
            stdout=verbosity,
            n=k,
            temperature=temperature
        )

        for j in range(k):
            code_blocks = ''.join(responses[j]['code_blocks'])
            output_data['index'].append(i)
            output_data['predict'].append(code_blocks)
            output_data['actual'].append(sample['output'])
            output_data['tests'].append(sample['tests'])

        if (i + 1) % batch_size == 0:
            pd.DataFrame(output_data).to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
            
            output_data = {
                'index': [],
                'predict': [],
                'actual': [],
                'tests': []
            }
            
    pd.DataFrame(output_data).to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
    

def generate_code_llama_code(
        model_name: str
) -> None:
    pass

if __name__ == '__main__':
    for dataset_name in dataset_names:
        dataset_location = os.path.join(dataset_root, dataset_name)

        if not os.path.exists(dataset_location):
            print("Unable to find", dataset_name, "at", dataset_location)
            continue

        if verbosity:
            print("Reading", dataset_name, "at", dataset_location)
        
        dataset_generator = get_dataset_generator(
            dataset_location=dataset_location,
            nrows=n_samples
        )
        
        if model_name in ['gpt-3.5-turbo', 'gpt-4']:
            generate_gpt_code(
                model_name=model_name,
                dataset_name=dataset_name,
                dataset_generator=dataset_generator
            )