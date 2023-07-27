'''
This file is use to generate alpaca dataset with gpt-3.5-turbo-0613 using (alfred) database.
Usage: python create_gpt_respond_db.py
'''
import os
import numpy as np
import json
# from prompt.prompt_utils import *
from prompt.prompt_utils_v4_db import PromptManager
import openai
import time
from tqdm import tqdm

# Set your own API
os.environ['OPENAI_API_KEY'] = "xxx"
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.Model.list())

with open("response.json", 'w') as f:
    json_str = json.dumps(openai.Model.list(), indent=2)
    f.write(json_str)
    f.write('\n')

json_file = "alpaca_6k.json"    # scene file
f = open(json_file, 'r')
content = f.read()
input_list = json.loads(content)
f.close()

input_list = input_list[:10]    # for scene test

# alfred_db_path = "prompt/alfred_trainset_database_6574.txt"   # full database
alfred_db_path = "prompt/alfred_trainset_database_100.txt"  # for database test
prompt_manager = PromptManager(alfred_db_path)
for input_one in tqdm(input_list):
    print("================================================")
    index = input_list.index(input_one)
    save_path = os.path.join(os.getcwd(), "respond_db", str(index) + ".json")
    instruction = input_one["instruction"]
    input_context = input_one["input"]
    if not instruction:
        input_context = "List of objects: " + input_context
    else:
        input_context = "List of objects: " + input_context + "\n" + "Instruction: " + instruction

    # TODO: use langchain to generate prompt
    messages = prompt_manager.get_prompt_as_ai2thor()

    # Add fewshot_samples
    # print(f"input_context: {input_context}")
    # samples = prompt_manager.get_fewshot_sample_db(input_context, k=1)
    samples = prompt_manager.get_fewshot_sample()
    for sample_one in samples:
        messages.append({"role": "user", "content": sample_one["context"]})
        messages.append({"role": "assistant", "content": sample_one["response"]})
    messages.append({"role": "user", "content": input_context})
    time.sleep(6)
    while True:
        try:
            print(f"messages: {messages}")
            response = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo-0301",
                # model="gpt-3.5-turbo-0613",
                # model="gpt-4-8k",
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.8,  # 0.0 - 2.0
                max_tokens=2048,
            )
            break
        except:
            time.sleep(20)

    print(f"response: {response['choices'][0]['message']}")
    with open(save_path, 'w') as f:
        json_str = json.dumps(response, indent=2)
        f.write(json_str)
        f.write('\n')
