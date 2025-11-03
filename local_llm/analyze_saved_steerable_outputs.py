import argparse
import json
import re
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score


# Answer format prompts and patterns
PATTERN = r"Answer:\s*([A-Za-z])"

PATTERN_COMPILED = re.compile(PATTERN)

# Stop sequences for the model output
STOP_SEQUENCE = ["\n\n# End of response"]

# Answer option mappings
ANSWER_OPTIONS = {"A": 0, "B": 1, "C": 2}

# Default answer option if format is invalid
ANSWER_DEFAULT = 0


def extract_answer(output: str, pattern: str) -> str:
    """
    Extracts the answer letter from the model output by looking for a pattern like \\boxed{A}
    Returns the extracted letter (uppercase) if found, else returns an empty string.
    """

    # Using a regex to capture a single uppercase letter inside 
    matches = re.findall(pattern, output)

    if matches:
        last_occurence = matches[-1]
        # Remove the surrounding whitespaces
        last_occurence = last_occurence.strip()
        return last_occurence.upper()

    return ""


def remove_last_answer(text: str) -> str:
    """
    Removes the last answer from the text based on the defined pattern, so that we have the cot only
    """
    matches = list(PATTERN_COMPILED.finditer(text))
    if not matches:
        return text  # nothing to remove

    # start / end indices of the final match
    start, end = matches[-1].span()
    return text[:start] + text[end:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file", type=str, default="./results/llama3_steerable_opinion_qa_grpo_v1_step_300_test.jsonl", help="Path to the answers JSON file containing ground-truth answers."
    )
    args = parser.parse_args()

    # Load the answers JSON file
    dataset = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line.strip()))

    pred = []
    gold = []

    # Check the answers
    for sample in dataset:
        index = sample["index"]
        output = sample.get("llm_output", "")
        cot = remove_last_answer(output)
        answer = extract_answer(output, PATTERN)

        # Store the result in the dataset
        sample["llm_output"] = output
        sample["cot"] = cot
        sample["predicted_answer"] = answer

        ground_truth = sample.get("correct_valence_option", "")
        sample["correct"] = answer == ground_truth

        pred.append(ANSWER_OPTIONS.get(answer, ANSWER_DEFAULT))  # Convert answer to index
        gold.append(ANSWER_OPTIONS.get(ground_truth, ANSWER_DEFAULT))


    print("Accuracy:", accuracy_score(gold, pred))
    print("Balanced accuracy:", balanced_accuracy_score(gold, pred))
    print("Macro F1 score:", f1_score(gold, pred, average="macro"))
    print("Micro F1 score:", f1_score(gold, pred, average="micro"))

    # binary setting where "either" options and predictions are removed
    pred_binary = []
    gold_binary = []
    for i in range(len(pred)):
        if gold[i] == 2:
            continue
        pred_binary.append(pred[i])
        gold_binary.append(gold[i])

    print("Binary accuracy:", accuracy_score(gold_binary, pred_binary))
    print("Binary balanced accuracy:", balanced_accuracy_score(gold_binary, pred_binary))
    print("Binary macro F1 score:", f1_score(gold_binary, pred_binary, labels=[0, 1], average="macro"))
    print("Binary micro F1 score:", f1_score(gold_binary, pred_binary, labels=[0, 1], average="micro"))
