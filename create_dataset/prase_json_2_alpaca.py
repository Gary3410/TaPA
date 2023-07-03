import os
import json
import numpy as np
import re
from tqdm import tqdm
import random

def prase_content_without_target(json_content):
    wrong_sig = -1
    # json_content_list = json_content.split("\n\n")
    if "===" not in json_content and "---\n\n" not in json_content:
        json_content = json_content.replace("\n\n", "===")
    json_content = json_content.replace("==", "===")
    json_content = json_content.replace("====", "===")
    content_list = []
    instruction_list = []
    actions_list = []

    json_content_list = json_content.split("===")
    # print(json_content_list)
    for json_content_one in json_content_list:
        try:
            json_content_one_list = json_content_one.split("---")
            json_content_one_instruction = json_content_one_list[0]
            json_content_one_actions = json_content_one_list[1]
            sp_instruction = json_content_one_instruction.split(": ")[1].rstrip()
            sp_actions = json_content_one_actions.split("tions:")[1].rstrip()

            # Iteration processing for each action
            p1 = re.compile(r'[(](.*?)[)]', re.S)
            sp_target = re.findall(p1, sp_actions)
            sp_target = ",".join(sp_target)

            if "None" in sp_target or "N/A" in sp_target or "if " in sp_target or "If " in sp_target:
                wrong_sig = 0
                continue
            sp_actions = re.sub(u"\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", sp_actions)
            sp_actions = sp_actions.replace(" \n", "\n")
            instruction_list.append(sp_instruction)
            actions_list.append(sp_actions)
        except:
            wrong_sig = 1
            continue

    if len(instruction_list) != len(actions_list):
        if wrong_sig == 0 or wrong_sig == 1:
            wrong_sig = wrong_sig
        else:
            wrong_sig = 2
        return [], wrong_sig

    for ind in range(len(instruction_list)):
        content_one_dict = {}
        content_one_dict["instruction"] = instruction_list[ind]
        if not actions_list[ind]:
            continue
        content_one_dict["output"] = actions_list[ind]
        content_list.append(content_one_dict)
    return content_list, wrong_sig


def create_dict():
    dict = {"instruction": [], "input": [], "output": [], "json_id": []}
    return dict


base_path = os.getcwd()
json_path = os.path.join(base_path, "respond")
json_list = os.listdir(json_path)

json_list.sort(key = lambda x: int(x[:-5]))
print(len(json_list))

json_file = "alpaca_6k.json"
f = open(json_file, 'r')
content = f.read()
input_list = json.loads(content)
f.close()

new_json_list = []
bad_name_list = []
format_wrong_list = []
content_wrong_list = []
save_path = "alpaca_15k_instruction.json"

kitchens_train = [f"FloorPlan{i}" for i in range(1, 21)]
living_rooms_train = [f"FloorPlan{200 + i}" for i in range(1, 21)]
bedrooms_train = [f"FloorPlan{300 + i}" for i in range(1, 21)]
bathrooms_train = [f"FloorPlan{400 + i}" for i in range(1, 21)]

scene_name_list = kitchens_train + living_rooms_train + bedrooms_train + bathrooms_train

for i in tqdm(range(len(json_list))):
    json_file_one = open(os.path.join(json_path, json_list[i]), "r")
    content = json_file_one.read()
    json_dict = json.loads(content)
    json_file_one.close()
    json_content = json_dict["choices"][0]["message"]["content"]

    context_list, wrong_sig = prase_content_without_target(json_content)

    if wrong_sig == 0 or wrong_sig == 1 or wrong_sig == 2:
        if wrong_sig == 0:
            content_wrong_list.append(json_list[i])
        else:
            format_wrong_list.append(json_list[i])

    input_ind = int(json_list[i][:-5])
    input_one = input_list[input_ind]
    input_one_content = input_one["input"]

    for context_list_one in context_list:
        new_dict = create_dict()
        new_dict["instruction"] = context_list_one["instruction"]
        new_dict["input"] = input_one_content
        new_dict["output"] = context_list_one["output"]
        # new_dict["scene_name"] = scene_name_list[i]
        new_dict["json_id"] = json_list[i]
        new_json_list.append(new_dict)


random.shuffle(new_json_list)
with open(save_path, 'w') as f:
    json_str = json.dumps(new_json_list, indent=2)
    f.write(json_str)
    f.write('\n')

print(len(new_json_list))
print("Number of formatting errors: ", len(format_wrong_list))
print("Number of content errors: ", len(content_wrong_list))
print("Total number of errors: ", len(format_wrong_list) + len(content_wrong_list))
print(format_wrong_list)