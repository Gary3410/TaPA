"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
import numpy as np

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm


DATA_FILE = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"
# DATA_FILE_NAME = "alpaca_data_cleaned_archive.json"
# DATA_FILE_NAME = "alpaca_4k_instruction.json"
DATA_FILE_NAME = "alpaca_15k_instruction.json"
IGNORE_INDEX = -1


def prepare(
    destination_path: Path = Path("data/alpaca"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    test_split_size: int = 600,
    max_seq_length: int = 512,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    data_file_name: str = DATA_FILE_NAME
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)
    file_path = destination_path / data_file_name
    # download(file_path)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    
    with open(file_path, "r") as file:
        data = json.load(file)

    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    """
    train_set, test_set = random_split(
        data, 
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)
    """
    train_set = data[:train_split_size]
    test_set = data[-test_split_size:]

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]

    len_list = []
    prompt_list = []
    for train_set_one in train_set:
        len_list.append(train_set_one["input_ids"].shape[0])
        if len_list[-1] == 256:
            prompt_list.append(generate_prompt(train_set_one) + train_set_one["output"])
    len_list = np.asarray(len_list)

    # If the number greater than 512 is too large,
    # you may need to increase the value of max_seq_length or modify the dataset
    print("Number of max_seq_lengths greater than 512：", np.sum(len_list >= 512))
    print("max_seq_length", np.max(len_list))
    print("min_seq_length", np.min(len_list))

    torch.save(train_set, file_path.parent / "train_15k.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, file_path.parent / "test_15k.pt")


def download(file_path: Path):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w") as f:
        f.write(requests.get(DATA_FILE).text)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # You can use the following three lines of code to check if the dataset makes sense
    # encoded_full_prompt = tokenize_without_max(tokenizer, full_prompt, max_length=max_length, eos=False)
    # encoded_full_prompt_and_response = tokenize_without_max(tokenizer, full_prompt_and_response,
    #                                                           eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def tokenize_without_max(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""
    # Original prompt
    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )
    # You might be able to modify the prompt to get better results
    # v1 : gives step-by-step guidance
    # v2 : list each step to finish the instruction

    # if example["input"]:
    #     return (
    #         "Below is an instruction that describes a task, paired with an input that provides further context. "
    #         "Write a response that appropriately completes the request and list each step to finish the instruction.\n\n"
    #         f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
    #     )
    # return (
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request and list each step to finish the instruction.\n\n"
    #     f"### Instruction:\n{example['instruction']}\n\n### Response:"
    # )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
