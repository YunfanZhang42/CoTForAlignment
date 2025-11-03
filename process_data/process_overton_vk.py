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
        "train": {},
        "val": {},
        "test": {}
    }

    processed_datasets_current_index = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    dataset = datasets.load_dataset("allenai/ValuePrism", "full", split="train")

    for i in dataset:
        situation = i["situation"]
        if situation not in situations_split:
            print(f"Situation '{situation}' not found in any split.")
            continue

        split = situations_split[situation]

        if situation not in processed_datasets[split]:
            processed_datasets[split][situation] = {
                "situation": situation,
                "vrds": [],
                "index": processed_datasets_current_index[split],
            }
            processed_datasets_current_index[split] += 1

        processed_datasets[split][situation]["vrds"].append({
            "text": i["text"],
            "explanation": i["explanation"],
            "valence": i["valence"],
        })

    for split in processed_datasets:
        for situation, item in processed_datasets[split].items():
            processed_datasets[split][situation]["all_explanations"] = "\n".join(f"- {i['text']}: {i['explanation']}" for i in item["vrds"])

    # Save the processed datasets
    for split, data in processed_datasets.items():
        output_file = os.path.join(args.output_dir, f"overton_vk_{split}.jsonl")
        with open(output_file, "w") as f:
            for item in data.values():
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} items to {output_file}")
