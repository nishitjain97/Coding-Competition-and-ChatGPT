"""
File used to run and evaluate GPT codes.

Nishit Jain, November 2023
"""

import os
import pandas as pd
from openai import OpenAI
import time
import json
import subprocess

from utils import *

## Set output location
root = "/Users/nishitjain/Documents/Nishit/Masters/AP/coding_competition_gpt/test/20231114_160059"

## Set flag to print updates
stdout = True

## Test case type
test_case_type = 'public'

if not os.path.exists(root):
    print(f"{root} does not exist. Please point to the correct directory.")
    exit()

print(f"Reading from {root}")

skip_dirs = ['.DS_Store']
stats = []

for dir in os.listdir(root):
    if dir in skip_dirs:
        continue

    inner_result = {
        'Problem_Index': dir,
        'N_Test_Cases': 0,
        'N_Passed': 0,
        'N_Failed': 0,
        'N_Timeouts': 0,
        'N_Errors': 0,
        'Test_Case_Type': test_case_type
    }

    prob_directory = os.path.join(root, dir)

    if 'gpt_code.py' not in os.listdir(prob_directory):
        print(f"GPT code not in {prob_directory}. Moving to next.")
        continue

    code_file = os.path.join(prob_directory, 'gpt_code.py')
    test_file = os.path.join(prob_directory, 'test_cases.txt')
    output_file = os.path.join(prob_directory, 'code_outputs.txt')
    temp_inp_file = os.path.join(prob_directory, 'temp_inp.txt')

    with open(test_file, 'r') as f:
        test_cases = json.load(f)

    ips = test_cases[test_case_type + '_input']
    ops = test_cases[test_case_type + '_output']
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    for ip, op in zip(ips, ops):
        inner_result['N_Test_Cases'] += 1

        # Write input to a file
        with open(temp_inp_file, 'w') as f:
            f.writelines(ip)
            
        try:
            command = "python3 " + code_file + " < " + temp_inp_file
            result = subprocess.run(command, shell=True, timeout=2, capture_output=True)
        except:
            inner_result['N_Timeouts'] += 1
            continue
        
        if result.returncode == 1:
            inner_result['N_Errors'] += 1
            
            with open(os.path.join(prob_directory, 'logs_' + timestr + '.txt'), 'a') as f:
                f.writelines(result.stderr.decode('utf-8') + "\n\n")
        else:
            if result.stdout.decode('utf-8').strip() == op.strip():
                inner_result['N_Passed'] += 1
            else:
                inner_result['N_Failed'] += 1
    stats.append(inner_result)

timestr = time.strftime("%Y%m%d_%H%M%S")
file_name = os.path.join(root, 'results_' + timestr + '.csv')

_ = pd.DataFrame(stats).to_csv(file_name)