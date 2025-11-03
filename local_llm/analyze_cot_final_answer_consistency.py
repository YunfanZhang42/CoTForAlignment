import argparse
import json
import random
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import VK_OVERTON_DATASET_NAME, VK_STEERABLE_DATASET_NAME, OPINION_QA_DATASET_NAME, get_dataset_name


# Answer format prompts and patterns
PATTERN = r"Answer:\s*([A-Za-z])"

PATTERN_COMPILED = re.compile(PATTERN)


VALENCE_MAP ={"Supports": "supports", "Opposes": "opposes", "Either": "either supports or opposes"}


def process_entry(prompt, args):
    with OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_url) as client:
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={"type": "text"},
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                frequency_penalty=0,
                presence_penalty=0,
            )
            result = completion.choices[0].message.content.strip()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="./results/llama3_steerable_opinion_qa_grpo_v1_step_300_test.jsonl", help="Directory to save the results")
    parser.add_argument(
        "--prompt-template-path",
        type=str,
        default="./prompt_templates/steerable_opinion_qa_cot_consistency_eval.txt",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="your_api_key",
        help="API key",
    )
    parser.add_argument("--openai-api-url", type=str, default="https://api.anthropic.com/v1", help="OpenAI compatible API URL")
    parser.add_argument("--max-concurrent-requests", type=int, default=32, help="The maximum number of concurrent requests to the API")
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet-20250219", help="Override the model name in the input JSON file")
    parser.add_argument("--temperature", type=float, default=0.0, help="The temperature for the completion")
    parser.add_argument("--max-tokens", type=int, default=768, help="The maximum number of tokens to generate")
    parser.add_argument("--top-p", type=float, default=0.95, help="The top-p value for nucleus sampling")
    parser.add_argument("--samples", type=int, default=1000, help="Field name for the chain of thought")
    parser.add_argument("--random-seed", type=int, default=164398, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load the answers JSON file
    dataset = []
    with open(args.input_file, "r") as f:
        dataset = [json.loads(line) for line in f.readlines()]

    # Sort the dataset by index
    dataset.sort(key=lambda x: x["index"])

    # Shuffle the dataset
    random.seed(args.random_seed)
    random.shuffle(dataset)

    # Limit the dataset to the specified number of samples
    if args.samples > 0:
        dataset = dataset[:args.samples]

    # Load the prompt template
    with open(args.prompt_template_path, "r") as f:
        prompt_template = f.read().strip()

    # Determine the dataset name
    dataset_name = get_dataset_name(args.prompt_template_path)

    prompts = {}
    for sample in dataset:
        # Only provide rationalizations for incorrect samples
        index = sample["index"]
        if dataset_name == VK_STEERABLE_DATASET_NAME:
            prompt = prompt_template.format(
                situation=sample["situation"],
                explanation=sample.get("cot", "").strip(),
            )
        elif dataset_name == OPINION_QA_DATASET_NAME:
            options_text = "\n".join([f"({i['option']}) {i['option_text']}" for i in sample["options"]])

            prompt = prompt_template.format(
                question_raw=sample["question_raw"],
                options_text=options_text,
                explanation=sample.get("cot", "").strip(),
            )
        else:
            raise NotImplementedError(f"Unknown dataset name: {dataset_name}")
        prompts[index] = prompt

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

    total_consistent = 0
    
    # Update the cot.
    for sample in dataset:
        index = sample["index"]
        output = responses[index]
        cot_verification_selected_answer = extract_answer(output, PATTERN_COMPILED)
        sample["cot_verification_selected_answer"] = cot_verification_selected_answer
        original_answer = sample.get("predicted_answer", "")
        sample["cot_verification_consistent"] = original_answer.upper() == cot_verification_selected_answer.upper()
        if sample["cot_verification_consistent"]:
            total_consistent += 1

    # Save the updated dataset
    file_name = args.input_file.split("/")[-1].replace(".jsonl", "_cot_consistency_result_sample.jsonl")
    with open(file_name, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Total samples: {len(dataset)}")
    print(f"Total consistent samples: {total_consistent}")
    print(f"Consistency rate: {total_consistent / len(dataset) * 100:.2f}%")

