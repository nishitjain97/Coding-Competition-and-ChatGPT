"""
File used to generate and store GPT codes.

Nishit Jain, November 2023

Updates:
    Nishit Jain, February 2024
"""

import os
import pandas as pd
from openai import OpenAI
import time
import json
import ast

from utils import *

## Set API key
api_key_location = "/Users/nishitjain/Documents/Nishit/Masters/AP/gpt_api_key.txt"
with open(api_key_location, 'r') as f:
    client = OpenAI(
        api_key=f.readlines()[0]
    )

## Root location
root_dir = "."

## Input Folder
## [Does not use root_dir. Give complete path]
problems_path = "/Users/nishitjain/Documents/Nishit/Masters/AP/collect_data/source_4_codecontests/0000.parquet"

## Output Folder
output_folder = "test"
output_location = os.path.join(root_dir, output_folder)

## Prompt template location
prompt_template_file = "prompt_template.txt"
prompt_template_loc = os.path.join(root_dir, prompt_template_file)

## Set flag to print updates
stdout = True

## Specify model
model = 'gpt-3.5-turbo'

if not os.path.exists(output_location):
    print(f"{output_location} does not exist. Please create directory before running this file.")
    exit()

timestr = time.strftime("%Y%m%d_%H%M%S")
root = os.path.join(output_location, timestr)
os.mkdir(root)
print(f"Storing results in {root}")

## Read problems to generate GPT codes
problems = pd.read_parquet(problems_path)

## Read prompt template
with open(prompt_template_loc, 'r') as f:
    prompt_template = ''.join(f.readlines())

for index in problems.index[:10]:
    problem = problems.loc[index]

    test_cases_public = problem['public_tests']
    test_cases_private = problem['private_tests']
    test_cases_generated = problem['generated_tests']

    if len(test_cases_public['output']) == 0:
        print("Problem", index, " - NO PUBLIC TEST CASES")
        continue
    
    if len(test_cases_private['output']) == 0:
        print("Problem", index, " - NO PRIVATE TEST CASES")
        continue

    prob_directory = os.path.join(root, str(index))
    os.mkdir(prob_directory)

    with open(os.path.join(prob_directory, 'test_cases.txt'), 'w') as f:
        f.writelines(json.dumps({
            'public_input': list(test_cases_public['input']),
            'public_output': list(test_cases_public['output']),
            'private_input': list(test_cases_private['input']),
            'private_output': list(test_cases_private['output']),
            'extra_input': list(test_cases_generated['input']),
            'extra_output': list(test_cases_generated['output'])
        }))

    if stdout:
        print(f"Problem index {index} stored at {prob_directory}")
        
    prompt = create_prompt(
        description={
            '%{prompt}%': problem['description']
        },
        prompt_template=prompt_template
    )

    with open(os.path.join(prob_directory, 'prompt.txt'), 'w') as f:
        f.writelines(prompt)

    full_response, code_blocks = ask_chatgpt(
        client=client,
        prompt=prompt,
        model=model,
        stdout=stdout
    )

    with open(os.path.join(prob_directory, 'gpt_response.txt'), 'w') as f:
        f.writelines(full_response)

    code_blocks = ''.join(code_blocks)

    with open(os.path.join(prob_directory, 'gpt_code.py'), 'w') as f:
        f.writelines(code_blocks)

    tree = ast.parse(code_blocks)
    with open(os.path.join(prob_directory, 'ast.txt'), 'w') as f:
        f.writelines(ast.dump(tree))