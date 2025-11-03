import argparse
import json
import random
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


def process_entry(prompt, args):
    with OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_url) as client:
        try:
            response = client.moderations.create(
                model=args.model,
                input=prompt,
            )
            
            result = response.results[0]
            return result.categories, result.category_scores

        except Exception as e:
            print(f"Error processing prompt:\n{prompt}\n{e}")

    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file", type=str, default="./results/llama3_steerable_opinion_qa_grpo_v1_step_300_test.jsonl", help="Path to the answers JSON file containing ground-truth answers."
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="your_api_key",
        help="API key",
    )
    parser.add_argument("--model", type=str, default="omni-moderation-2024-09-26", help="Model to use for the API")
    parser.add_argument("--openai-api-url", type=str, default="https://api.openai.com/v1", help="OpenAI compatible API URL")
    parser.add_argument("--max-concurrent-requests", type=int, default=256, help="The maximum number of concurrent requests to the API")
    parser.add_argument("--samples", type=int, default=2000, help="Maximum number of tokens to generate")
    parser.add_argument("--random-seed", type=int, default=164398, help="Random seed for reproducibility")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        dataset = [json.loads(line) for line in f.readlines()]

    # Sort the dataset by index
    dataset.sort(key=lambda x: x["index"])

    # Set the random seed for reproducibility
    random.seed(args.random_seed)
    random.shuffle(dataset)

    # Limit the dataset to the specified number of samples
    if args.samples > 0:
        dataset = dataset[:args.samples]

    responses = {}
    # Use ThreadPoolExecutor for concurrency
    with ThreadPoolExecutor(max_workers=args.max_concurrent_requests) as executor:
        futures = {executor.submit(process_entry, data["cot"].strip(), args): data["index"] for data in dataset}
        for future in as_completed(futures):
            try:
                idx = futures[future]
                responses[idx] = future.result()
            except Exception as e:
                print(f"Error processing entry: {e}")

    # Save a copy of the dataset with violations
    dataset_with_violations = []
    for idx in dataset:
        if idx["index"] in responses:
            violations, scores = responses[idx["index"]]
            if violations is None or scores is None:
                continue
            idx["openai_moderation_violations"] = dict(violations)
            dataset_with_violations.append(idx)

    file_name = args.input_file.split("/")[-1].replace(".jsonl", "_openai_moderation_result_sample.jsonl")
    with open(file_name, "w") as f:
        for entry in dataset_with_violations:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    overall_violations_count = 0
    per_category_violations = defaultdict(int)
    per_category_scores_sum = defaultdict(float)
    overall_scores_sum = 0.0
    categories = set()

    total_entries = len(responses)
    if total_entries == 0:
        print("No data to aggregate.")
        exit()

    for idx in responses:
        violations, scores = responses[idx]
        violations = dict(violations)
        scores = dict(scores)

        if violations is None or scores is None:
            continue  # Skip entries with errors

        # Check for any violation to count overall
        if any(violations.values()):
            overall_violations_count += 1

        # Update per-category violations
        for category, violated in violations.items():
            if violated:
                per_category_violations[category] += 1

        # Update per-category scores and overall sum
        for category, score in scores.items():
            per_category_scores_sum[category] += score
            overall_scores_sum += score
            categories.add(category)

    # Calculate overall metrics
    overall_violation_ratio = overall_violations_count / total_entries
    num_categories = len(categories) if categories else 1
    overall_avg_score = overall_scores_sum / (total_entries * num_categories) if total_entries * num_categories > 0 else 0.0

    # Calculate per-category metrics
    per_category_violation_ratio = {cat: count / total_entries for cat, count in per_category_violations.items()}
    per_category_avg_score = {cat: per_category_scores_sum[cat] / total_entries for cat in categories}

    # Print results
    print("Aggregated Results:")
    print(f"Overall Violation Ratio: {overall_violation_ratio:.2%}")
    print(f"Overall Average Score: {overall_avg_score:.4f}\n")

    print("Per Category Results:")
    for category in sorted(categories):
        vr = per_category_violation_ratio.get(category, 0.0)
        avg = per_category_avg_score.get(category, 0.0)
        print(f"Category: {category}")
        print(f"  Violation Ratio: {vr:.2%}")
        print(f"  Average Score: {avg:.4f}")

    
