# Convert the dataset into parquet format as required by verl.
import argparse
import os
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from common import VK_OVERTON_DATASET_NAME, VK_STEERABLE_DATASET_NAME, OPINION_QA_DATASET_NAME, get_dataset_name, format_templates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="./datasets/overton_vk_val_star_small.jsonl", help="Path to the input JSON file.")
    parser.add_argument("--template-file", type=str, default="./prompt_templates/overton_vk_zs_eval.txt", help="Path to the template JSON file.")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Tokenizer to use to estimate the input length.")
    args = parser.parse_args()

    # Load the input JSON file
    data = []
    with open(args.input_file, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Load the template file
    with open(args.template_file, "r") as f:
        template = f.read().strip()

    # Get the dataset name
    dataset_name = get_dataset_name(args.template_file)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    input_lengths = []
    transformed_data = []

    for i in data:
        index = i["index"]

        _, prompt = format_templates(dataset_name, "", template, "", i)

        # Tokenize the prompt to get the input length
        tokenized = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_lengths.append(tokenized.shape[-1])
        
        if dataset_name == VK_STEERABLE_DATASET_NAME or dataset_name == OPINION_QA_DATASET_NAME:
            ground_truth_answer = i["correct_valence_option"]
        elif dataset_name == VK_OVERTON_DATASET_NAME:
            ground_truth_answer = i["final_comment"]

        temp_data = {
            "data_source": "default",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "default",
            "reward_model": {"style": "rule", "ground_truth": ground_truth_answer},
            "extra_info": {"split": "default", "index": i["index"]},
        }

        if dataset_name == VK_OVERTON_DATASET_NAME:
            temp_data["extra_info"]["explanations"] = [j["explanation"] for j in i["vrds"]]

        transformed_data.append(temp_data)

    # Convert transformed_data to a pandas DataFrame
    df = pd.DataFrame(transformed_data)

    # Save the DataFrame as a parquet file
    output_file = os.path.splitext(args.input_file)[0] + ".parquet"
    df.to_parquet(output_file)

    print(f"Average number of valid tokens: {np.mean(input_lengths)}")
    print(f"Standard deviation of valid tokens: {np.std(input_lengths)}")
    print(f"Minimum number of valid tokens: {np.min(input_lengths)}")
    print(f"Maximum number of valid tokens: {np.max(input_lengths)}")

    print(f"Top 0.1% of valid tokens: {np.percentile(input_lengths, 99.99)}")
    print(f"Top 1% of valid tokens: {np.percentile(input_lengths, 99)}")
    print(f"Top 5% of valid tokens: {np.percentile(input_lengths, 95)}")
    print(f"Top 10% of valid tokens: {np.percentile(input_lengths, 90)}")
