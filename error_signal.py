# External Libraries
import pandas as pd
import os
import json
import subprocess

from trace_values import trace_main
from utils import generate_tree

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

if __name__ == "__main__":
    for dataset_name in dataset_names:
        file_location = os.path.join(output_root, 'results_first_iteration')
        
        file_path = os.path.join(file_location, f"{dataset_name.replace('/', '-')}_{model_name}.csv")

        if not os.path.exists(file_path):
            print(file_path, "does not exist.")
            continue

        dataset = pd.read_csv(file_path)

        dataset['compiler_error'] = '-'
        dataset['ast'] = '-'
        dataset['variable_trace'] = '-'
        dataset['error_doc'] = '-'
        dataset['flake8'] = '-'
        dataset['timeout_flag'] = False

        for index, row in dataset.iterrows():
            llm_code = row['predict']

            with open('code_tmp.py', 'w') as f:
                f.writelines(llm_code)

            test_cases = json.loads(row['tests'])['public_tests']

            inputs = test_cases['input']

            with open('inp_tmp.txt', 'w') as f:
                f.writelines(inputs)

            timeout_flag = False

            try:
                command = "python3 code_tmp.py < inp_tmp.txt"
                result = subprocess.run(command, shell=True, timeout=2, capture_output=True)
            except:
                timeout_flag = True
                
            if not timeout_flag and result.returncode == 0:
                continue
            elif timeout_flag:
                dataset.loc[index, 'timeout_flag'] = timeout_flag
                continue