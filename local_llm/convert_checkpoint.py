import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-model-path", type=str, default="./checkpoints/llama3_steerable_vk_human_cot_latest.pth", help="Path to your PyTorch-saved model")
    parser.add_argument("--hf-model-type", type=str, default="Qwen/Qwen2.5-7B", help="Hugging Face model type")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_type)
    model = AutoModelForCausalLM.from_pretrained(args.hf_model_type)

    # Load the PyTorch model
    model.load_state_dict(torch.load(args.torch_model_path))

    model = model.to(torch.bfloat16)

    # Save path is the same without extension
    if args.torch_model_path.endswith("_best.pth"):
        hf_save_path = args.torch_model_path.replace("_best.pth", "")
    elif args.torch_model_path.endswith("_latest.pth"):
        hf_save_path = args.torch_model_path.replace("_latest.pth", "")
    else:
        raise ValueError("Invalid model path. It should end with '_best.pth' or '_latest.pth'.")

    # Save the model in Hugging Face format
    tokenizer.save_pretrained(hf_save_path)
    model.save_pretrained(hf_save_path)

    print(f"Model successfully converted and saved to {hf_save_path}")