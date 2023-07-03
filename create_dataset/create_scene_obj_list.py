import os
import random

import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import cv2
from tqdm import tqdm

# Train
kitchens_train = [f"FloorPlan{i}" for i in range(1, 21)]
living_rooms_train = [f"FloorPlan{200 + i}" for i in range(1, 21)]
bedrooms_train = [f"FloorPlan{300 + i}" for i in range(1, 21)]
bathrooms_train = [f"FloorPlan{400 + i}" for i in range(1, 21)]

#  Val
kitchens_val = [f"FloorPlan{i}" for i in range(21, 26)]
living_rooms_val = [f"FloorPlan{200 + i}" for i in range(21, 26)]
bedrooms_val = [f"FloorPlan{300 + i}" for i in range(21, 26)]
bathrooms_val = [f"FloorPlan{400 + i}" for i in range(21, 26)]

# Test
kitchens_test = [f"FloorPlan{i}" for i in range(26, 31)]
living_rooms_test = [f"FloorPlan{200 + i}" for i in range(26, 31)]
bedrooms_test = [f"FloorPlan{300 + i}" for i in range(26, 31)]
bathrooms_test = [f"FloorPlan{400 + i}" for i in range(26, 31)]

scenes_train = kitchens_train + living_rooms_train + bedrooms_train + bathrooms_train
scenes_val = kitchens_val + living_rooms_val + bedrooms_val + bathrooms_val
scenes_test = kitchens_test + living_rooms_test + bedrooms_test + bathrooms_test

controller = Controller(
            agentMode="default",
            visibilityDistance=100,
            scene="FloorPlan319",
            platform=CloudRendering,
            makeAgentsVisible=False,

            # step sizes
            gridSize=0.25,
            snapToGrid=True,
            rotateStepDegrees=90,
            farClippingPlane=200,

            # image modalities
            renderDepthImage=True,
            renderInstanceSegmentation=True,

            # camera properties
            width=1440,
            height=1440,
            fieldOfView=90
        )

def mkdir_file(mode, scene_one):
    base_path = os.getcwd()
    rgb_file = os.path.join(base_path, "dataset")
    if not os.path.exists(os.path.join(rgb_file, mode)):
        os.makedirs(os.path.join(rgb_file, mode))
    train_save_file = os.path.join(rgb_file, mode, scene_one)
    if not os.path.exists(train_save_file):
        os.makedirs(train_save_file)
    return train_save_file


def save_box_list(save_path, obj_name_list, obj_box_list):
    f = open(os.path.join(save_path, "name_box.txt"), "w")
    box_list = []
    for ind in range(len(obj_name_list)):
        obj_name_one = obj_name_list[ind]
        obj_box_one = obj_box_list[ind]
        obj_box_one = np.asarray(obj_box_one)
        try:
            obj_box_one = np.around(obj_box_one, 2)
            box_list.append(obj_name_one + ": " + str(obj_box_one.tolist()))
        except:
            box_list.append(obj_name_one + ": " + "None")
        # box_list.append(obj_name_one + ": " + str(obj_box_one))
    l = ""
    for z in box_list:
        l += "{}".format(z) + ", "
    f.write(l.strip(", "))
    f.close()


def save_name_list(save_path, obj_name_list):
    object_name_list_save = obj_name_list.copy()
    # object_name for de-duplication
    object_name_unique = list(set(object_name_list_save))
    f = open(os.path.join(save_path, "name.txt"), "w")
    l = ""
    for z in object_name_unique:
        l += "{}".format(z) + ", "
    f.write(l.strip(", "))
    f.close()


def save_name_position_list(save_path, obj_name_list, obj_position_list):
    f = open(os.path.join(save_path, "name_position.txt"), "w")
    name_position_list = []
    for ind in range(len(obj_name_list)):
        obj_name_one = obj_name_list[ind]
        obj_position_one = list(obj_position_list[ind].values())
        try:
            obj_position_one = [round(x, 2) for x in obj_position_one]
            name_position_list.append(obj_name_one + ": " + str(obj_position_one))
        except:
            name_position_list.append(obj_name_one + ": " + "None")
    l = ""
    for z in name_position_list:
        l += "{}".format(z) + ", "
    f.write(l.strip(", "))
    f.close()


def save_top_down_image(save_path, top_down_frames):
    rgb_save_path = os.path.join(save_path, "top_down.png")
    frame_one = top_down_frames[0]
    frame_one = frame_one[:, :, :3]
    frame_one = frame_one[:, :, ::-1]
    cv2.imwrite(rgb_save_path, frame_one)


def main():
    mode = "train"
    # mode = "val"
    # 20 for val and 80 for train
    scene_num = 80
    for i in tqdm(range(scene_num)):
        scene_one = scenes_train[int(i)]
        for j in tqdm(range(1)):

            scene_save_one = scene_one + "_" + str(j)
            save_file = mkdir_file(mode, scene_save_one)
            controller.reset(scene=scene_one)
            last_event = controller.last_event
            obj_metadata = last_event.metadata["objects"]
            obj_name_list = []
            obj_box_list = []
            obj_position_list = []
            for obj_ind in range(len(obj_metadata)):
                obj_name = obj_metadata[obj_ind]["objectId"].split("|")[0]
                visible = obj_metadata[obj_ind]["visible"]
                if obj_name != "Floor":
                    obj_name_list.append(obj_name)
                    obj_box_list.append(obj_metadata[obj_ind]["axisAlignedBoundingBox"]["cornerPoints"])
                    obj_position_list.append(obj_metadata[obj_ind]["position"])

            save_box_list(save_file, obj_name_list, obj_box_list)
            save_name_list(save_file, obj_name_list)
            save_name_position_list(save_file, obj_name_list, obj_position_list)

            # Get top_down view
            event = controller.step(action="GetMapViewCameraProperties")
            event = controller.step(
                action="AddThirdPartyCamera",
                **event.metadata["actionReturn"]
            )

            top_down_frames = event.third_party_camera_frames
            save_top_down_image(save_file, top_down_frames)

if __name__ == '__main__':
    main()