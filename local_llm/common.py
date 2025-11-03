from typing import List, Dict, Any, Tuple


VK_OVERTON_DATASET_NAME = "VK_OVERTON"
VK_STEERABLE_DATASET_NAME = "VK_STEERABLE"
OPINION_QA_DATASET_NAME = "OpinionQA"


def get_dataset_name(template_path: str) -> str:
    """
    Returns the dataset name based on the template path.
    """
    if "overton_vk" in template_path.lower():
        return VK_OVERTON_DATASET_NAME
    elif "steerable_vk" in template_path.lower():
        return VK_STEERABLE_DATASET_NAME
    elif "opinion_qa" in template_path.lower():
        return OPINION_QA_DATASET_NAME
    else:
        raise NotImplementedError(f"Unknown dataset name.")


def format_templates(dataset_name: str, train_template: str, eval_template: str, cot_field: str, sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    Formats the template with the provided values.
    """
    train_text = ""
    eval_text = ""

    if dataset_name == VK_OVERTON_DATASET_NAME:
        if train_template:
            train_text = train_template.format(
                situation=sample["situation"],
                explanation=sample.get(cot_field, ""),
                final_comment=sample["final_comment"],
            )
        if eval_template:
            eval_text = eval_template.format(
                situation=sample["situation"],
                explanation=sample.get(cot_field, ""),
            )
    elif dataset_name == VK_STEERABLE_DATASET_NAME:
        if train_template:
            train_text = train_template.format(
                    situation=sample["situation"],
                    vrd_text=sample["vrd_text"],
                    explanation=sample.get(cot_field, ""),
                    correct_valence_option= sample["correct_valence_option"],
            )
        if eval_template:
            eval_text = eval_template.format(
                situation=sample["situation"],
                vrd_text=sample["vrd_text"],
            )
    elif dataset_name == OPINION_QA_DATASET_NAME:
        options_text = "\n".join([f"({i['option']}) {i['option_text']}" for i in sample["options"]])
        if train_template:
            train_text = train_template.format(
                question_raw=sample["question_raw"],
                attribute_natural_language=sample["attribute_natural_language"],
                group=sample["group"],
                options_text=options_text,
                explanation=sample.get(cot_field, ""),
                correct_valence_option=sample["correct_valence_option"],
            )
        if eval_template:
            eval_text = eval_template.format(
                question_raw=sample["question_raw"],
                attribute_natural_language=sample["attribute_natural_language"],
                group=sample["group"],
                options_text=options_text,
            )
    else:
        raise NotImplementedError(f"Unknown dataset name: {dataset_name}")
    
    return train_text, eval_text