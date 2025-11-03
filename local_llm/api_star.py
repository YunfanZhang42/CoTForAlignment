import argparse
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import VK_OVERTON_DATASET_NAME, VK_STEERABLE_DATASET_NAME, OPINION_QA_DATASET_NAME, get_dataset_name


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        default="./datasets/overton_vk_test.jsonl",
        help="Path to the answers JSON file containing ground-truth answers.",
    )
    parser.add_argument("--output-file", type=str, default="./results/llama3_steerable_vk_human_cot_test.jsonl", help="Directory to save the results")
    parser.add_argument(
        "--prompt-template-path",
        type=str,
        default="./prompt_templates/star_overton_vk.txt",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="your_api_key",
        help="API key",
    )
    parser.add_argument("--openai-api-url", type=str, default="https://api.openai.com/v1", help="OpenAI compatible API URL")
    parser.add_argument("--max-concurrent-requests", type=int, default=1000, help="The maximum number of concurrent requests to the API")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini-2025-04-14", help="Override the model name in the input JSON file")
    parser.add_argument("--temperature", type=float, default=0.7, help="The temperature for the completion")
    parser.add_argument("--max-tokens", type=int, default=768, help="The maximum number of tokens to generate")
    parser.add_argument("--top-p", type=float, default=0.95, help="The top-p value for nucleus sampling")
    args = parser.parse_args()

    # Load the answers JSON file
    dataset = []
    with open(args.input_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    # Load the prompt template
    with open(args.prompt_template_path, "r") as f:
        prompt_template = f.read().strip()

    # Determine the dataset name
    dataset_name = get_dataset_name(args.prompt_template_path)

    prompts = {}
    for sample in dataset:
        # Only provide rationalizations for incorrect samples
        if not sample.get("correct", False):
            index = sample["index"]
            if dataset_name == VK_OVERTON_DATASET_NAME:
                prompt = prompt_template.format(
                    situation=sample["situation"],
                    explanation=sample["all_explanations"],
                )
            elif dataset_name == VK_STEERABLE_DATASET_NAME:
                prompt = prompt_template.format(
                    situation=sample["situation"],
                    vrd_text=sample["vrd_text"],
                    valence=VALENCE_MAP.get(sample["valence"], "either supports or opposes"),
                    explanation=sample["explanation"],
                )
            elif dataset_name == OPINION_QA_DATASET_NAME:
                options_text = "\n".join([f"({i['option']}) {i['option_text']}" for i in sample["options"]])

                # Find the correct option text
                correct_option_text = ""
                for i in sample["options"]:
                    if i["option"] == sample["correct_valence_option"]:
                        correct_option_text = i["option_text"]
                        break

                prompt = prompt_template.format(
                    question_raw=sample["question_raw"],
                    attribute_natural_language=sample["attribute_natural_language"],
                    group=sample["group"],
                    options_text=options_text,
                    correct_valence_option=sample["correct_valence_option"],
                    correct_valence_option_text=correct_option_text,
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

    # Update the cot.
    for sample in dataset:
        index = sample["index"]

        if dataset_name == VK_OVERTON_DATASET_NAME:
            sample["final_comment"] = responses[index]
        else:
            if index in responses:
                sample["cot"] = responses[index]
            else:
                sample["cot"] = sample["cot"].strip()

            sample.pop("llm_output", None)
            sample.pop("predicted_answer", None)
            sample.pop("correct", None)

    # Save the results
    with open(args.output_file, "w") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
