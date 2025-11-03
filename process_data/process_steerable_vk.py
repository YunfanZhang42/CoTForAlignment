import argparse
import os
import json
import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render LaTeX HF dataset questions to PNG")
    parser.add_argument("--output-dir", type=str, default="./datasets/", help="Directory to the dataset.")
    args = parser.parse_args()

    # Load the dataset
    situations_split = {}

    for split in ["train", "val", "test"]:
        dataset = datasets.load_dataset("allenai/ValuePrism", "explanation", split=split)
        for i in dataset:
            situation = i["situation"]
            if situation in situations_split and situations_split[situation] != split:
                print(f"Situation '{situation}' found in multiple splits: {situations_split[situation]} and {split}")
                # Make sure test has the highest priority, and then val, then train
                if situations_split[situation] == "test" or split == "test":
                    situations_split[situation] = "test"
                elif situations_split[situation] == "val" or split == "val":
                    situations_split[situation] = "val"
                print(f"Situation '{situation}' set to split: {situations_split[situation]}")
            else:
                situations_split[situation] = split

    processed_datasets = {
        "train": [],
        "val": [],
        "test": []
    }

    processed_datasets_current_index = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    options = {
        "Supports": "A",
        "Opposes": "B",
        "Either": "C",
    }

    dataset = datasets.load_dataset("allenai/ValuePrism", "full", split="train")

    for i in dataset:
        situation = i["situation"]
        if situation not in situations_split:
            print(f"Situation '{situation}' not found in any split.")
            continue

        split = situations_split[situation]

        processed_datasets[split].append({
            "situation": situation,
            "vrd": i["vrd"],
            "vrd_text": i["text"],
            "explanation": i["explanation"],
            "valence": i["valence"],
            "correct_valence_option": options[i["valence"]],
            "index": processed_datasets_current_index[split]
        })

        processed_datasets_current_index[split] += 1
    
    # Save the processed datasets
    for split, data in processed_datasets.items():
        output_file = os.path.join(args.output_dir, f"steerable_vk_{split}.jsonl")
        with open(output_file, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} items to {output_file}")
