import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from scripts.prepare_alpaca import generate_prompt

# Detic的依赖
import mss
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo
import multiprocessing as mp
import argparse
import os
import json
import random
from tqdm import tqdm
import numpy as np
import math


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    # cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = '/home/wzy/Detic/datasets/metadata/our_coco_clip_a+cname.npy'
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def create_dict():
    dict = {"instruction": [], "input": [], "input_GT": [], "output": [], "output_GT": [], "scene_name": []}
    return dict


def mkdir_scene_file(scene_path):
    if not os.path.exists(scene_path):
        os.makedirs(scene_path)


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--navigation_strategy",
        default="Priori_partial_60",
        choices=['Traversal_0.25_60', 'Traversal_0.25_120', 'Traversal_0.75_60',
                 'Traversal_0.75_120', 'Priori_partial_60', 'Priori_overall_60'],
        help="",
    )
    parser.add_argument(
        "--base_data_path",
        default="./ai2thor/dataset",
        help="",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"],
        nargs=argparse.REMAINDER,
    )
    return parser


def main(
    prompt: str = "Please tidy up the room",
    input: str = "",
    adapter_path: Path = Path("out/adapter/alpaca/lit-llama-adapter-finetuned_15k.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = "llm.int8",
    max_new_tokens: int = 1024,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    # First start Detic
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args)

    # The next step is to start lit-llama
    assert adapter_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Loading model ...", file=sys.stderr)
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ):
            model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model.load_state_dict(adapter_checkpoint, strict=False)

    model.eval()
    model = fabric.setup_module(model)
    tokenizer = Tokenizer(tokenizer_path)

    # Start random cruise mode
    # Configure the base file path
    mode = args.navigation_strategy
    base_path = args.base_data_path
    position_dict_base_path = os.path.join(base_path, mode, "position_dict")
    rgb_img_base_path = os.path.join(base_path, mode, "rgb_img")
    out_filename_base_path = os.path.join(os.getcwd(), "Visualization_results")

    # Get the scene to be selected
    kitchens_val = [f"FloorPlan{i}" for i in range(21, 26)]
    living_rooms_val = [f"FloorPlan{200 + i}" for i in range(21, 26)]
    bedrooms_val = [f"FloorPlan{300 + i}" for i in range(21, 26)]
    bathrooms_val = [f"FloorPlan{400 + i}" for i in range(21, 26)]

    # Get the validation label
    f = open("./data/alpaca/alpaca_20_val_instruction.json", 'r')
    content = f.read()
    prompt_dict = json.loads(content)
    f.close()

    # Set save file
    base_save_path = os.getcwd()
    save_path = os.path.join(base_save_path, "out", mode)
    mkdir_scene_file(save_path)
    # Get the file path
    scene_name_list = kitchens_val + living_rooms_val + bedrooms_val + bathrooms_val
    image_num_list = []
    for scene_one_name in scene_name_list:

        f = open(os.path.join(position_dict_base_path, scene_one_name + ".json"), 'r')
        content = f.read()
        position_dict = json.loads(content)
        f.close()

        filtered_list = [d for d in prompt_dict if d.get("scene_name") == scene_one_name]

        degree_step = int(mode.split("_")[-1])
        if degree_step == 60:
            view_id_list = list(range(0, 6))  # D=60
        else:
            view_id_list = list(range(0, 6, 2))  # D=120
        position_list = position_dict.copy()

        label_list = []
        for position_ind, position_one in tqdm(position_list.items()):
            for random_view_id in view_id_list:
                rgb_image_name = scene_one_name + "_" + str(position_ind) + "_" + str(random_view_id) + ".png"
                img_path = os.path.join(rgb_img_base_path, rgb_image_name)
                img = read_image(img_path, format="BGR")

                predictions, visualized_output = demo.run_on_image(img)

                metadata = demo.metadata
                instances = predictions["instances"].to("cpu")
                classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
                class_names = metadata.get("thing_classes", None)
                if class_names is not None and len(class_names) > 0:
                    labels = [class_names[i] for i in classes]
                    label_list.extend(labels)

                # Save visualization results
                # out_filename = os.path.join(out_filename_base_path, rgb_image_name)
                # visualized_output.save(out_filename)


        label_list = list(set(label_list))
        if len(label_list) > 80:
            label_list = random.sample(label_list, 80)
        input = ", ".join(label_list)
        input = "[" + input + "]"

        save_list = []
        for instruction_dict_one in tqdm(filtered_list):
            prompt = instruction_dict_one["instruction"]
            sample = {"instruction": prompt, "input": input}
            prompt = generate_prompt(sample)
            encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

            y = generate(
                model,
                idx=encoded,
                max_seq_length=max_new_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_id=tokenizer.eos_id
            )

            output = tokenizer.decode(y)
            output = output.split("### Response:")[1].strip()

            new_dict = create_dict()
            new_dict["instruction"] = instruction_dict_one["instruction"]
            new_dict["input"] = input
            new_dict["output"] = output
            new_dict["input_GT"] = instruction_dict_one["input"]
            new_dict["output_GT"] = instruction_dict_one["output"]
            new_dict["scene_name"] = instruction_dict_one["scene_name"]
            new_dict["images_num"] = len(position_list) * len(view_id_list)
            save_list.append(new_dict)

        image_num_list.append(len(position_list) * len(view_id_list))

        save_path_one = os.path.join(save_path, scene_one_name + ".json")
        with open(save_path_one, 'w') as f:
            json_str = json.dumps(save_list, indent=2)
            f.write(json_str)
            f.write('\n')

    print("The average number of images collected: ", sum(image_num_list) / len(image_num_list))
if __name__ == "__main__":
    # from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    # CLI(main)
    main()
