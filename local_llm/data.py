import argparse
import json
import os

import torch
import numpy as np
from dotmap import DotMap
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from common import get_dataset_name, format_templates


HF_LOSS_IGNORE_TOKEN_ID = -100


class MCCoTDataset(Dataset):
    def __init__(self, dataset_path, tokenizer=None, context_length=1024, 
                 train_template_path="", eval_template_path="", 
                 mode="train", cot_field="explanation"):
        super().__init__()

        self.tokenizer = tokenizer
        self.context_length = context_length

        # Set up padding and truncation options for the tokenizer
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.mode = mode
        self.cot_field = cot_field

        # HACK: use template name to determine the dataset name
        self.dataset_name = get_dataset_name(train_template_path)

        # Load the templates.
        with open(train_template_path, "r", encoding="utf-8") as f:
            self.train_template = f.read().strip()

        with open(eval_template_path, "r", encoding="utf-8") as f:
            self.eval_template = f.read().strip()

        self.data = []

        # Read the dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                self.data.append(sample)
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Randomly select an article
        sample = self.data[idx]        
        
        train_text, eval_text = format_templates(
            dataset_name=self.dataset_name,
            train_template=self.train_template,
            eval_template=self.eval_template,
            cot_field=self.cot_field,
            sample=sample,
        )

        text = train_text if self.mode == "train" or self.mode == "val" else eval_text

        data = self.tokenizer(text, max_length=self.context_length, truncation=True, padding="max_length", return_tensors="pt")

        data["full_text"] = text

        # Remove the first dimension of the input_ids, attention_mask.
        data["input_ids"] = data["input_ids"].squeeze(0)
        data["attention_mask"] = data["attention_mask"].squeeze(0)

        if self.mode == "train":
            labels = data["input_ids"].clone()

            # Tokenize just the input part to find where it ends
            input_tokens = self.tokenizer.encode(eval_text, max_length=self.context_length, truncation=True, padding=False, add_special_tokens=True)
            
            # Mask the labels for the input part
            labels[0: len(input_tokens)] = HF_LOSS_IGNORE_TOKEN_ID
            
            # Mask the labels for the padding tokens
            labels[data["attention_mask"] == 0] = HF_LOSS_IGNORE_TOKEN_ID

            data["labels"] = labels

        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test loading natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./configs/llama3_overton_vk_star.json", help="Path to config file")
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        args = DotMap(json.load(f))

    # Test and load the train set
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=True)
    train_dataset = MCCoTDataset(
        dataset_path=os.path.join(args.dataset_dir, args.train_filename),
        tokenizer=tokenizer,
        context_length=args.max_context_length,
        train_template_path=args.train_template_path,
        eval_template_path=args.eval_template_path,
        mode="train",
        cot_field=args.cot_field,
    )
    torch.set_printoptions(threshold=10_000)

    # Print the first few examples
    for i in range(5):
        # print(train_dataset[i])
        # Translate the token IDs back to text, including the special tokens
        print(tokenizer.decode(train_dataset[i]["input_ids"]))
        print("Calculate loss for the following tokens:")
        print(tokenizer.decode(train_dataset[i]["labels"][train_dataset[i]["labels"] != HF_LOSS_IGNORE_TOKEN_ID]))
        print("-" * 50)

    print("Length of train set:", len(train_dataset))

    # Iterate over the whole train set and print the statistics of valid, not padded tokens
    valid_tokens = []
    for i in range(len(train_dataset)):
        valid_tokens.append(train_dataset[i]["attention_mask"].sum().item())

    print(f"Average number of valid tokens: {np.mean(valid_tokens)}")
    print(f"Standard deviation of valid tokens: {np.std(valid_tokens)}")
    print(f"Minimum number of valid tokens: {np.min(valid_tokens)}")
    print(f"Maximum number of valid tokens: {np.max(valid_tokens)}")

    # Print the percentage of samples with >= max context length tokens
    print(f"Percentage of samples exceeding max context length tokens: {np.mean(np.array(valid_tokens) >= args.max_context_length) * 100}")

    # Print the top 1% of valid tokens
    print(f"Top 1% of valid tokens: {np.percentile(valid_tokens, 99)}")
    # Print the top 5% of valid tokens
    print(f"Top 5% of valid tokens: {np.percentile(valid_tokens, 95)}")
    # Print the top 10% of valid tokens
    print(f"Top 10% of valid tokens: {np.percentile(valid_tokens, 90)}")
