import os
import numpy as np
import json
import cv2
from tqdm import tqdm
import random


def create_dict():
    dict = {"instruction": [], "input": [], "output": []}
    return dict


def get_obj_list(scene_name_all, scene_name_list_one):

    diff_list = list(set(scene_name_all).difference(set(scene_name_list_one)))
    scene_name_list_one_copy = scene_name_list_one.copy()
    change_number = random.randint(1, int(len(scene_name_list_one_copy) / 2))
    random.shuffle(scene_name_list_one_copy)
    if change_number < len(diff_list):
        new_obj_name_list = scene_name_list_one_copy[:-change_number] + random.sample(diff_list, change_number)
    else:
        new_obj_name_list = scene_name_list_one_copy[:-len(diff_list)] + diff_list

    if len(scene_name_list_one_copy) <= 40:
        change_number = random.randint(int(len(scene_name_list_one_copy) / 2), len(scene_name_list_one_copy))
    else:
        change_number = random.randint(int(len(scene_name_list_one_copy) / 2), 40)
    new_obj_name_list = random.sample(new_obj_name_list, change_number)

    return new_obj_name_list


base_path = os.getcwd()
data_path = os.path.join(base_path, "dataset", "train")
scene_file_list = os.listdir(data_path)

scene_name_list = []

kitchens_train = [f"FloorPlan{i}" for i in range(1, 21)]
living_rooms_train = [f"FloorPlan{200 + i}" for i in range(1, 21)]
bedrooms_train = [f"FloorPlan{300 + i}" for i in range(1, 21)]
bathrooms_train = [f"FloorPlan{400 + i}" for i in range(1, 21)]

scene_name_list.append(kitchens_train)
scene_name_list.append(living_rooms_train)
scene_name_list.append(bedrooms_train)
scene_name_list.append(bathrooms_train)


save_path = "alpaca_6k.json"
json_list = []
for scene_name_one in tqdm(scene_name_list):
    scene_name_all = []
    scene_name_all_list = []
    for scene_one in scene_name_one:
        scene_one_path = os.path.join(data_path, scene_one + "_" + str(0))
        scene_one_name_txt = os.path.join(scene_one_path, "name.txt")
        with open(scene_one_name_txt, "r") as tf:
            name_list_one = tf.read().split(',')
        scene_name_all.extend(name_list_one)
        scene_name_all_list.append(name_list_one)

    scene_name_all = list(set(scene_name_all))
    for scene_ind in range(len(scene_name_one)):
        for i in range(80):
            new_obj_list = get_obj_list(scene_name_all, scene_name_all_list[scene_ind])
            new_dict = create_dict()
            new_obj_list = ",".join(new_obj_list)
            new_obj_list = "[" + new_obj_list.strip() + "]"
            new_dict["input"] = new_obj_list.strip()
            json_list.append(new_dict)

with open(save_path, 'w') as f:
    json_str = json.dumps(json_list, indent=2)
    f.write(json_str)
    f.write('\n')





