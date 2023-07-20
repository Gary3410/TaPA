import os
import numpy as np
import json
from prompt.prompt_utils import *
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

json_file = "alpaca_6k.json"
f = open(json_file, 'r')
content = f.read()
input_list = json.loads(content)
f.close()

for input_one in tqdm(input_list[1:]):
    index = input_list.index(input_one)
    save_path = os.path.join(os.getcwd(), "respond", str(index) + ".json")
    instruction = input_one["instruction"]
    input_context = input_one["input"]
    if not instruction:
        input_context = "List of objects: " + input_context + "\n" + "Instruction: " + instruction
    else:
        input_context = "List of objects: " + input_context

    messages = get_prompt()
    # Add fewshot_samples
    samples = get_fewshot_sample()
    for sample_one in samples:
        messages.append({"role": "user", "content": sample_one["context"]})
        messages.append({"role": "assistant", "content": sample_one["response"]})
    messages.append({"role": "user", "content": input_context})
    time.sleep(6)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=messages,
                temperature=0.8,  # 0.0 - 2.0
                max_tokens=2048,
            )
            break
        except:
            time.sleep(20)

    with open(save_path, 'w') as f:
        json_str = json.dumps(response, indent=2)
        f.write(json_str)
        f.write('\n')
