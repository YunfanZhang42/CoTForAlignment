import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    PATTERN = r"Answer:\s*([A-Z])$"
    CORRECT_REWARD = 1.0
    INCORRECT_REWARD = 0.0

    # Using a regex to capture a single uppercase letter inside 
    matches = re.findall(PATTERN, solution_str)

    if matches:
        last_occurence = matches[-1]
        # Remove the surrounding whitespaces
        last_occurence = last_occurence.strip()
        if last_occurence == ground_truth:
            return CORRECT_REWARD
    
    return INCORRECT_REWARD
