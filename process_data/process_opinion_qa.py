import argparse
import glob
import os
import random
import json
import re
import pandas as pd


ATTRIBUTE_TO_NATURAL_LANGUAGE = {
    "AGE": "Age:",
    "CITIZEN": "US Citizen:",
    "CREGION": "US Census Region:",
    "EDUCATION": "Education Level:",
    "INCOME": "Income Range:",
    "MARITAL": "Marital Status:",
    "Overall": "General US Population",
    "POLIDEOLOGY": "Political Ideology:",
    "POLPARTY": "Political Party Affiliation:",
    "RACE": "Race:",
    "RELIG": "Religion:",
    "RELIGATTEND": "Religious Attendance:",
    "SEX": "Gender:",
}

ALL_CAPITAL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def save_jsonl(data, file_path):
    """
    Save a list of dictionaries to a JSONL file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process opinion_qa datasets.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/yunfan/Workspace/opinions_qa/data/distributions",
        help="Folder containing American_Trends_Panel_WXX_default_combined.csv files.",
    )
    parser.add_argument("--output-dir", type=str, default="./datasets/", help="Output directory for processed dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Proportion of data to use for training.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Proportion of data to use for validation.")
    parser.add_argument("--random-seed", type=int, default=908164, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.random_seed)

    # Find all CSV files matching the pattern
    pattern = os.path.join(os.path.abspath(args.input_dir), "American_Trends_Panel_W*_default_combined.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No CSVs found with pattern\n  {pattern}. Check --input-dir or file names."
        )

    # Read every file into a DataFrame and add a `source` column indicating the survey name
    frames = [pd.read_csv(path).assign(source=os.path.basename(path).split("_default_combined.csv")[0])for path in paths]

    # Concatenate all DataFrames into one
    df = pd.concat(frames, ignore_index=True, sort=False)

    # Quick report
    print(f"Loaded {len(paths)} CSV file(s):")
    for p in paths:
        print(os.path.basename(p))
    print("Columns available:")
    for c in df.columns:
        print(c)
    print(f"Total rows across all CSVs: {len(df):,}")

    # Keep only the relevant columns and rename D_H -> human_opinion_distribution
    columns_to_keep = ["question_raw", "ordinal_refs", "attribute", "group", "D_H", "source"]
    df = df[columns_to_keep].rename(columns={"D_H": "human_opinion_distribution"}).copy()

    # If both "attribute" and "group" are "Overall", set "group" to ""
    df.loc[(df["attribute"] == "Overall") & (df["group"] == "Overall"), "group"] = ""

    # Read "ordinal_refs" as a list of strings and "human_opinion_distribution" as a list of floats
    df["ordinal_refs_lists"] = df["ordinal_refs"].apply(lambda x: eval(x))
    df["human_opinion_distribution_lists"] = df["human_opinion_distribution"].apply(lambda x: eval(re.sub(r"\s+", ",", x)))

    # View final DataFrame shape after trimming and renaming
    print("After selecting key columns and renaming:")
    print(df.head())

    # Sanity check: Ensure consistency of human_opinion_distribution within identical question/group combinations
    dup_counts = df.groupby(["question_raw", "ordinal_refs", "attribute", "group", "source",])["human_opinion_distribution"].nunique()
    inconsistencies = dup_counts[dup_counts > 1]

    if not inconsistencies.empty:
        print("Inconsistencies detected in opinion distributions:")
        print(inconsistencies.to_string())
        raise ValueError(
            "Inconsistent human_opinion_distribution values detected across identical question/group combinations."
        )
    else:
        print("All opinion distributions are consistent for each question/group combination.")

    # Keep only the unique combinations of question_raw, ordinal_refs, attribute, group.
    # Note that different sources may have different distributions for the same question/group.
    # When that happens, we will keep the human_opinion_distribution from the latest source.
    df = df.sort_values("source").reset_index(drop=True)

    df = df.drop_duplicates(subset=["question_raw", "ordinal_refs", "attribute", "group"], keep="last").reset_index(drop=True)

    num_entries = len(df)
    num_unique_questions = df["question_raw"].nunique()
    unique_attributes = sorted(df["attribute"].dropna().unique())

    print("Dataset summary statistics:")
    print(f"Total entries: {num_entries:,}")
    print(f"Unique questions: {num_unique_questions:,}")
    print(f"Unique attributes {len(unique_attributes)}:")
    for attr in unique_attributes:
        print(f"{attr}")

    # Replace ordial_refs and human_opinion_distribution with lists
    df["ordinal_refs"] = df["ordinal_refs_lists"]
    df = df.drop(columns=["ordinal_refs_lists"])
    df["human_opinion_distribution"] = df["human_opinion_distribution_lists"]
    df = df.drop(columns=["human_opinion_distribution_lists"])

    records = df.to_dict(orient="records")

    for i in records:
        i["attribute_natural_language"] = ATTRIBUTE_TO_NATURAL_LANGUAGE[i["attribute"]]
        i["options"] = []

        # Create options based on ordinal_refs and human_opinion_distribution
        for j, option_text in enumerate(i["ordinal_refs"]):
            i["options"].append({
                "option": ALL_CAPITAL_LETTERS[j],
                "option_text": option_text,
                "probability": i["human_opinion_distribution"][j]
            })

        # Set the correct_valence_option based on the option letter with the highest probability
        max_prob_index = i["human_opinion_distribution"].index(max(i["human_opinion_distribution"]))
        i["correct_valence_option"] = ALL_CAPITAL_LETTERS[max_prob_index]

    # Split the dataset into train, val, and test based on "question_raw"
    all_questions = df["question_raw"].unique()
    train_size = int(args.train_ratio * len(all_questions))
    val_size = int(args.val_ratio * len(all_questions))

    random.shuffle(all_questions)

    train_questions = all_questions[:train_size]
    val_questions = all_questions[train_size:train_size + val_size]
    test_questions = all_questions[train_size + val_size:]

    train_set = [i for i in records if i["question_raw"] in train_questions]
    val_set = [i for i in records if i["question_raw"] in val_questions]
    test_set = [i for i in records if i["question_raw"] in test_questions]

    # Assign an integer index to each set
    for i in [train_set, val_set, test_set]:
        for j in i:
            j["index"] = i.index(j)

    # Save the datasets to JSONL files
    save_jsonl(train_set, os.path.join(args.output_dir, "steerable_opinion_qa_train.jsonl"))
    save_jsonl(val_set, os.path.join(args.output_dir, "steerable_opinion_qa_val.jsonl"))
    save_jsonl(test_set, os.path.join(args.output_dir, "steerable_opinion_qa_test.jsonl"))
