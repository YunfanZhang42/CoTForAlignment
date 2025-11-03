# This script calls an OpenAI compatible API to evaluate a model.
# First convert PyTorch checkpoints to Hugging Face format using `convert_checkpoint.py` (if not already).
# Then, run vllm to serve the model (if local model), e.g.:
# vllm serve ./checkpoints/llama3_steerable_vk_human_cot --tokenizer meta-llama/Meta-Llama-3-8B --host localhost --port 8001 --data-parallel-size 8 --max-model-len 1K

import argparse
import json
import re
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from common import VK_OVERTON_DATASET_NAME, VK_STEERABLE_DATASET_NAME, OPINION_QA_DATASET_NAME, get_dataset_name, format_templates


# Answer format prompts and patterns
PATTERN = r"Answer:\s*([A-Za-z])"

PATTERN_COMPILED = re.compile(PATTERN)

# Answer format for overton tasks
OVERTON_PATTERN_COMPILED = re.compile(r"Final Comment:(.*)", re.DOTALL)

# Stop sequences for the model output
STOP_SEQUENCE = ["\n\n# End of response"]

# Answer option mappings
ANSWER_OPTIONS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9}

# Default answer option if format is invalid
ANSWER_DEFAULT = 0


def process_entry(prompt, args):
    with OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_url) as client:
        try:
            if args.instruct_model:
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    response_format={"type": "text"},
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                result = completion.choices[0].message.content.strip()
            else:
                completion = client.completions.create(
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=STOP_SEQUENCE,
                )
                result = completion.choices[0].text.strip()

        except Exception as e:
            print(f"Error processing prompt:\n{prompt}\n{e}")
            result = ""

    return result


def extract_answer(output: str, pattern_compiled: re.Pattern[str]) -> str:
    """
    Extracts the answer from the model output by looking for a pattern like \\boxed{A}
    Returns the extracted answer if found, else returns an empty string.
    """

    # Using a regex to capture a the answer
    matches = re.findall(pattern_compiled, output)

    if matches:
        last_occurence = matches[-1]
        # Remove the surrounding whitespaces
        last_occurence = last_occurence.strip()
        return last_occurence

    return ""


def remove_last_answer(text: str, pattern_compiled: re.Pattern[str]) -> str:
    """
    Removes the last answer from the text based on the defined pattern, so that we have the cot only
    """
    matches = list(pattern_compiled.finditer(text))
    if not matches:
        return text  # nothing to remove

    # start / end indices of the final match
    start, end = matches[-1].span()
    return text[:start] + text[end:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file", type=str, default="./datasets/steerable_opinion_qa_test.jsonl", help="Path to the answers JSON file containing ground-truth answers."
    )
    parser.add_argument("--output-file", type=str, default="./results/qwen2p5_steerable_opinion_qa_grpo_v2_global_step_180_test.jsonl", help="Directory to save the results")
    parser.add_argument(
        "--prompt-template-path",
        type=str,
        default="./prompt_templates/steerable_opinion_qa_zs_eval.txt",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="your_api_key",
        help="API key",
    )
    # For local vLLM deployment, use http://localhost:8001/v1
    # For OpenAI API, use https://api.openai.com/v1
    parser.add_argument("--openai-api-url", type=str, default="http://localhost:8001/v1", help="OpenAI compatible API URL")
    parser.add_argument("--max-concurrent-requests", type=int, default=2000, help="The maximum number of concurrent requests to the API")
    parser.add_argument("--model", type=str, default="./checkpoints/llama3_steerable_opinion_qa_grpo_v2_global_step_180", help="Override the model name in the input JSON file")
    parser.add_argument("--temperature", type=float, default=0.7, help="The temperature for the completion")
    parser.add_argument("--max-tokens", type=int, default=512, help="The maximum number of tokens to generate")
    parser.add_argument("--top-p", type=float, default=0.95, help="The top-p value for nucleus sampling")
    parser.add_argument("--instruct-model", action="store_true", help="Use instruct model format for the API")
    args = parser.parse_args()

    # Load the answers JSON file
    dataset = []
    with open(args.input_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    # Load the prompt template
    with open(args.prompt_template_path, "r") as f:
        prompt_template = f.read().strip()

    dataset_name = get_dataset_name(args.prompt_template_path)

    prompts = {}
    for sample in dataset:
        index = sample["index"]
        _, eval_prompt = format_templates(dataset_name=dataset_name, train_template="", eval_template=prompt_template, 
                                          cot_field="no_cot", sample=sample)
        prompts[index] = eval_prompt

    responses = {}
    # Use ThreadPoolExecutor for concurrency
    with ThreadPoolExecutor(max_workers=args.max_concurrent_requests) as executor:
        futures = {executor.submit(process_entry, prompt, args): idx for idx, prompt in prompts.items()}
        for future in as_completed(futures):
            try:
                idx = futures[future]
                responses[idx] = future.result()
            except Exception as e:
                print(f"Error processing entry: {e}")

    pred = []
    gold = []

    # Check the answers
    for sample in dataset:
        index = sample["index"]
        output = responses.get(index, "")
        pattern_to_use = OVERTON_PATTERN_COMPILED if dataset_name == VK_OVERTON_DATASET_NAME else PATTERN_COMPILED
        cot = remove_last_answer(output, pattern_to_use)
        answer = extract_answer(output, pattern_to_use)

        # Store the result in the dataset
        sample["llm_output"] = output
        sample["cot"] = cot

        if dataset_name == VK_STEERABLE_DATASET_NAME or dataset_name == OPINION_QA_DATASET_NAME:
            sample["predicted_answer"] = answer

            ground_truth = sample.get("correct_valence_option", "")
            sample["correct"] = answer.upper() == ground_truth.upper()

            pred.append(ANSWER_OPTIONS.get(answer, ANSWER_DEFAULT))  # Convert answer to index
            gold.append(ANSWER_OPTIONS.get(ground_truth, ANSWER_DEFAULT))

        elif dataset_name == VK_OVERTON_DATASET_NAME:
            # Make the dataset format compatible with modular pluralism evaluation script.
            sample["output"] = answer
            sample["explanation"] = [j["explanation"] for j in sample["vrds"]]

    if dataset_name == VK_STEERABLE_DATASET_NAME or dataset_name == OPINION_QA_DATASET_NAME:
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

    # Save the results
    with open(args.output_file, "w") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
