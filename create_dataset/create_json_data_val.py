import os
import numpy as np
import json
import cv2
from tqdm import tqdm
import random


def create_dict():
    dict = {"instruction": [], "input": [], "output": []}
    return dict


base_path = os.getcwd()
data_path = os.path.join(base_path, "rgb_frames", "val")
scene_file_list = os.listdir(data_path)

scene_name_list = []

kitchens_val = [f"FloorPlan{i}" for i in range(21, 26)]
living_rooms_val = [f"FloorPlan{200 + i}" for i in range(21, 26)]
bedrooms_val = [f"FloorPlan{300 + i}" for i in range(21, 26)]
bathrooms_val = [f"FloorPlan{400 + i}" for i in range(21, 26)]

scene_name_list.append(kitchens_val)
scene_name_list.append(living_rooms_val)
scene_name_list.append(bedrooms_val)
scene_name_list.append(bathrooms_val)


save_path = "alpaca_20_val.json"
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
        new_dict = create_dict()
        new_obj_list = ",".join(scene_name_all_list[scene_ind])
        new_obj_list = "[" + new_obj_list.strip() + "]"
        new_dict["input"] = new_obj_list.strip()
        json_list.append(new_dict)

with open(save_path, 'w') as f:
    json_str = json.dumps(json_list, indent=2)
    f.write(json_str)
    f.write('\n')





