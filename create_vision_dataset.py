import os
import numpy as np
import cv2
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from tqdm import tqdm
import json

# Train
kitchens_train = [f"FloorPlan{i}" for i in range(1, 21)]
living_rooms_train = [f"FloorPlan{200 + i}" for i in range(1, 21)]
bedrooms_train = [f"FloorPlan{300 + i}" for i in range(1, 21)]
bathrooms_train = [f"FloorPlan{400 + i}" for i in range(1, 21)]

# Val
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
# scenes_val = scenes_val + scenes_test

print("Number of training scenes: ", len(scenes_train))
print("Number of validation scenes: ", len(scenes_val))


controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene="FloorPlan319",
            platform=CloudRendering,
            makeAgentsVisible=False,

            # step sizes
            gridSize=0.75,
            snapToGrid=True,
            rotateStepDegrees=90,

            # image modalities
            renderDepthImage=True,
            renderInstanceSegmentation=True,

            # camera properties
            width=1024,
            height=1024,
            fieldOfView=90
        )


def save_rgb_frame(rgb_image, base_path, save_name):
    save_path = os.path.join(base_path, save_name + ".png")
    cv2.imwrite(save_path, rgb_image)


def save_depth_frame(depth_img, base_path, save_name):
    save_path = os.path.join(base_path, save_name + ".npy")
    depth_img = np.asarray(depth_img)
    np.save(save_path, depth_img)


def save_json_dict(json_file, base_path, save_name):
    save_path = os.path.join(base_path, save_name + ".json")
    with open(save_path, 'w') as f:
        json_str = json.dumps(json_file, indent=2)
        f.write(json_str)
        f.write('\n')


def mkdir_dataset_file(scene_path):
    if not os.path.exists(scene_path):
        os.makedirs(scene_path)


def create_dataset(scene_list, mode=None):
    base_path = os.getcwd()
    bad_scene = []
    rgb_base_path = os.path.join(base_path, "ai2thor", "dataset", mode, "rgb_img")
    position_dict_base_path = os.path.join(base_path, "ai2thor", "dataset", mode, "position_dict")
    mkdir_dataset_file(rgb_base_path)
    mkdir_dataset_file(position_dict_base_path)
    grid_size = float(mode.split("_")[1])
    if "random" in mode.lower():
        grid_size = 0.75
    for scene_one in tqdm(scene_list):
        # The simulator resets the scene
        try:
            controller.reset(scene=scene_one, gridSize=grid_size)
        except:
            bad_scene.append(scene_one)
            continue
        # Start BFS navigation module
        positions = controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]

        degree_step = 60
        position_dict = {}
        for position_one in positions:
            pos_id = positions.index(position_one)
            position_dict[str(pos_id)] = position_one
            event = controller.step(action='Teleport', position=position_one)
            for i in range(int(360 / degree_step)):
                event = controller.step(action="RotateRight", degrees=degree_step)
                last_event = controller.last_event
                rgb_image = last_event.cv2img
                # File naming rules: scene_name + position_id + view_id.png
                # position_id: scene_name + .json
                image_file_save_name = scene_one + "_" + str(pos_id) + "_" + str(i)
                position_dict_save_name = scene_one
                save_rgb_frame(rgb_image, rgb_base_path, image_file_save_name)
        save_json_dict(position_dict, position_dict_base_path, position_dict_save_name)

    # Some bugs in the AI2THOR simulator?
    print(bad_scene)
    print(len(bad_scene))

def main():
    # mode_list = ["train", "val"]
    mode_list = ["Random_1_60", "Random_1_120", "Random_75_60",
                 "Random_75_120", "Traversal_0.25_60", "Traversal_0.25_120",
                 "Traversal_0.75_60", "Traversal_0.75_120"]
    for mode_one in mode_list:
        if mode_one == "train":
            create_dataset(scenes_train, mode_one)
        else:
            create_dataset(scenes_val, mode_one)




if __name__ == '__main__':
    main()

