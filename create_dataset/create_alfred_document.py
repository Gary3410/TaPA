'''
This file is used to create the database file alfred_trainset_database.txt from alfred dataset.
To run this file, you need to access the processed data from alfred dataset first.
Usage: python create_dataset/create_alfred_document.py
'''
import os
import numpy as np
import json
# import orjson as json
import cv2
from tqdm import tqdm
import random


def create_dict():
    dict = {"instruction": '', "input": '', "output": ''}
    return dict

base_path = os.getcwd()
data_path = os.path.join(base_path, "create_dataset", "prompt")

def load_traj(scene_name):
    '''load the processed data from alfred dataset, please refer to the original repo for data processing'''
    json_dir = 'alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(
        scene_name['repeat_idx']) + '.json'
    traj_data = json.load(open(json_dir))
    return traj_data

scene_names_list = json.load(open('alfred/data/splits/oct21.json'))['train']

# Add a filter to do few-shot learning
print("filtering the data...")
scene_names_list = [scene_name for scene_name in scene_names_list if scene_name['repeat_idx'] == 0] # filter the repeat_idx
# TODO: remove short trajectories and add a filter to remove similiar trajectories
# scene_names_list = scene_names_list[:100] # for fast test
print("filtering done!, the number of samples is: ", len(scene_names_list)) # 6.5k

json_list = []

cnt = 0 # count the number of samples
for scene_name in tqdm(scene_names_list):
    sample = create_dict()
    traj_data = load_traj(scene_name)
    r_idx = traj_data['repeat_idx']

    object_list = [op['objectName'].split('_')[0] for op in traj_data['scene']['object_poses']]
    action_list = []
    for hl_idx, hl_action in enumerate(traj_data['plan']['high_pddl']):
        if hl_action['discrete_action']['action'] == 'NoOp' or len(hl_action['discrete_action']['args']) == 0 or \
                    hl_action['discrete_action']['args'][0] == '':
            break
        action_list.append(hl_action['discrete_action']['action']+"("+hl_action['discrete_action']['args'][0]+")")
    
    sample['instruction'] = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
    sample['input'] = '['+', '.join(object_list)+']'
    sample['output'] = ', '.join(action_list)
    json_list.append(sample)
    cnt += 1

save_path = os.path.join(data_path, "alfred_trainset_database_" + str(cnt) + ".txt")
with open(save_path, 'w') as f:
    # write each sample in json_list to a txt_str, split by \n\n
    txt_str = ''
    for sample in json_list:
        txt_str += json.dumps(sample) + '\n\n'
    
    # remove the last \n\n
    txt_str = txt_str[:-2]
    f.write(txt_str)



