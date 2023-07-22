import os
import numpy as np
import cv2
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from tqdm import tqdm
import json
from sklearn.cluster import KMeans

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
print("Number of val scenes:", len(scenes_val))


controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene="FloorPlan319",
            platform=CloudRendering,
            makeAgentsVisible=False,

            # step sizes
            gridSize=0.75,
            snapToGrid=False,
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


def get_overall_center(positions):
    overall_center_list = []
    overall_center = {}
    xs1 = np.asarray([p["x"] for p in positions])
    zs1 = np.asarray([p["z"] for p in positions])
    ys1 = np.asarray([p["y"] for p in positions])
    mean_x = np.mean(xs1)
    mean_z = np.mean(zs1)
    mean_y = np.mean(ys1)

    overall_center["x"] = mean_x
    overall_center["y"] = mean_y
    overall_center["z"] = mean_z
    overall_center_list.append(overall_center)

    return overall_center_list


def get_partial_center(positions):
    partial_center_list = []
    xs1 = np.asarray([p["x"] for p in positions])
    zs1 = np.asarray([p["z"] for p in positions])
    ys1 = np.asarray([p["y"] for p in positions])

    # x_z = np.concatenate((xs1, zs1), axis=1)
    x_z = np.asarray(list(zip(xs1, zs1)))
    optimal_clusters = find_optimal_clusters(x_z)
    if optimal_clusters is None:
        optimal_clusters = x_z.shape[0]
    kmeans = KMeans(n_clusters=optimal_clusters, n_init="auto")
    kmeans.fit(x_z)
    # labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    for i, centroid in enumerate(centroids):
        overall_center_one = {}
        overall_center_one["x"] = centroid[0]
        overall_center_one["y"] = np.mean(ys1)
        overall_center_one["z"] = centroid[1]
        partial_center_list.append(overall_center_one)
    return partial_center_list


def find_slow_trend_index(data):
    # Calculate the differences between consecutive elements
    differences = [data[i] - data[i - 1] for i in range(1, len(data))]

    # Find the index where the trend of differences becomes slower
    slow_index = None
    max_diff = None

    for i in range(1, len(differences)):
        current_diff = differences[i]
        prev_diff = differences[i - 1]

        if max_diff is None or current_diff < max_diff:
            max_diff = current_diff
        elif current_diff >= max_diff:
            slow_index = i
            break

    return slow_index + 1 if slow_index is not None else None


def find_optimal_clusters(points):
    # Calculate the within-cluster sum of squares (WCSS) for different numbers of clusters
    wcss = []
    max_clusters = len(points)  # Set the maximum number of clusters to the number of points

    for num_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
        kmeans.fit(points)
        wcss.append(kmeans.inertia_)

    # Find the optimal number of clusters
    # wcss_diff = abs(np.diff(wcss))
    optimal_clusters = find_slow_trend_index(np.diff(wcss))
    return optimal_clusters


def create_dataset(scene_list, mode=None):
    base_path = os.getcwd()
    bad_scene = []
    rgb_base_path = os.path.join(base_path, "dataset", mode, "rgb_img")
    depth_base_path = os.path.join(base_path, "dataset", mode, "depth_img")
    instance_base_path = os.path.join(base_path, "dataset", mode, "instance_img")
    position_dict_base_path = os.path.join(base_path, "dataset", mode, "position_dict")
    instance_dict_base_path = os.path.join(base_path, "dataset", mode, "instance_dict")
    for scene_one in tqdm(scene_list):

        try:
            controller.reset(scene=scene_one)
        except:
            bad_scene.append(scene_one)
            continue

        positions = controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]

        degree_step = 60

        if mode == "val_0.75_overall_center":
            positions_list = get_overall_center(positions)
        elif mode == "val_0.75_partial_center_vis":
            positions_list = get_partial_center(positions)
        else:
            positions_list = positions
        position_dict = {}
        for position_one in positions_list:
            pos_id = positions_list.index(position_one)
            position_dict[str(pos_id)] = position_one
            event = controller.step(action='Teleport', position=position_one)
            # event = controller.step(action='TeleportFull',  x=position_one["x"], y=position_one["y"], z=position_one["z"])
            for i in tqdm(range(int(360 / degree_step))):
                event = controller.step(action="RotateRight", degrees=degree_step)
                last_event = controller.last_event

                rgb_image = last_event.cv2img
                depth_image = last_event.depth_frame
                instance_segmentation_frame = last_event.instance_segmentation_frame
                color_to_object_id = last_event.color_to_object_id

                image_file_save_name = scene_one + "_" + str(pos_id) + "_" + str(i)
                position_dict_save_name = scene_one
                save_rgb_frame(rgb_image, rgb_base_path, image_file_save_name)
                # save_depth_frame(depth_image, depth_base_path, image_file_save_name)
                # save_rgb_frame(instance_segmentation_frame, instance_base_path, image_file_save_name)
                color_to_object_id_save = {str(k): color_to_object_id[k] for k in color_to_object_id}
        # save_json_dict(color_to_object_id_save, instance_dict_base_path, position_dict_save_name)
        save_json_dict(position_dict, position_dict_base_path, position_dict_save_name)
    print(bad_scene)
    print(len(bad_scene))

def main():
    # mode_list = ["train", "val"]
    # mode_list = ["val_0.75_overall_center", "val_0.75_partial_center"]
    mode_list = ["val_0.75_partial_center_vis"]
    for mode_one in mode_list:
        if mode_one == "train":
            create_dataset(scenes_train, mode_one)
        else:
            create_dataset(scenes_val, mode_one)




if __name__ == '__main__':
    main()

