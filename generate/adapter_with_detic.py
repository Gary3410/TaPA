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

# Detic dependence
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
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
    # cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = '/home/wzy/Detic/datasets/metadata/our_coco_clip_a+cname.npy'
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def create_dict():
    dict = {"instruction": [], "input": [], "input_GT": [], "output": [], "output_GT": [], "scene_name": []}
    return dict


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
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"],
        nargs=argparse.REMAINDER,
    )

    # New arguments
    parser.add_argument(
        "--prompt",
        default="Can you give me a book?",
        help="Tasks given by the user",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--img_path",
        default="./input/rgb_img",
        help="Scene images save path",
    )

    return parser


def main(
        prompt: str = "Please tidy up the room",
        input: str = "",
        adapter_path: Path = Path("out/adapter/alpaca/lit-llama-adapter-finetuned_15k.pth"),
        pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
        tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
        img_path: Path = Path("input/rgb_img"),
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
    # Base path
    out_filename_base_path = "./output/rgb_img"  # Replace with path to dataset

    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args)

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

    # Read image file
    img_path = args.img_path
    max_new_tokens = args.max_new_tokens
    print(img_path)
    print(max_new_tokens)
    img_list = os.listdir(img_path)
    if len(img_list) == 0:
        print("Warning, No image input")

    label_list = []
    for img_one in img_list:
        img_one_path = os.path.join(img_path, img_one)
        img = read_image(img_one_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        # Detection results
        metadata = demo.metadata
        instances = predictions["instances"].to("cpu")
        classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
        class_names = metadata.get("thing_classes", None)
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
            label_list.extend(labels)
        # The visualization can be saved using the following code
        out_filename = os.path.join(out_filename_base_path, img_one)
        visualized_output.save(out_filename)

    # Integrate scene information
    label_list = list(set(label_list))
    # Excessively long inputs can cause failures
    if len(label_list) > 80:
        label_list = random.sample(label_list, 80)
    input = ", ".join(label_list)
    input = "[" + input + "]"

    # Task planning generation
    prompt = args.prompt
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    print("===================")
    print(prompt)
    print("===================")
    print(f"token shape: {encoded.shape}", file=sys.stderr)
    prompt_length = encoded.size(0)

    t0 = time.perf_counter()
    y = generate(
        model,
        idx=encoded,
        max_seq_length=max_new_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id
    )
    t = time.perf_counter() - t0

    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()
    print(output)

    tokens_generated = y.size(0) - prompt_length
    print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    # CLI(main)
    main()
