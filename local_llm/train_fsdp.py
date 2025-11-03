import argparse
import time
import random
import os
import json
import gc
import functools

from dotmap import DotMap
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from data import MCCoTDataset


def setup(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.master_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def save_model(model, rank, path, model_name):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
        if rank == 0:
            torch.save(state_dict, os.path.join(path, model_name + ".pth"))
            print(f"Saved model to {os.path.join(path, model_name + '.pth')}")


def main(rank, world_size, args):
    setup(rank, world_size, args)
    print(f"Initialized process group with rank {rank} and world size {world_size}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Finished setting up random seeds and TF32 on rank {rank}")

    # Set up the tensorboard on rank 0
    if rank == 0:
        log_writer = SummaryWriter(os.path.join(args.tensorboard_log_dir, args.model_name))

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    print(f"Finished initializing tokenizer on rank {rank}")

    train_dataset = MCCoTDataset(
        dataset_path=os.path.join(args.dataset_dir, args.train_filename),
        tokenizer=tokenizer,
        context_length=args.max_context_length,
        train_template_path=args.train_template_path,
        eval_template_path=args.eval_template_path,
        mode="train",
        cot_field=args.cot_field,
    )

    val_dataset = MCCoTDataset(
        dataset_path=os.path.join(args.dataset_dir, args.val_filename),
        tokenizer=tokenizer,
        context_length=args.max_context_length,
        train_template_path=args.train_template_path,
        eval_template_path=args.eval_template_path,
        mode="train",
        cot_field=args.cot_field,
    )

    # Initialize the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.gradient_accumulation_steps // world_size,
        shuffle=False,
        # Training LLMs will be slow anyway, so we don't need to use multiple workers
        num_workers=1,
        sampler=DistributedSampler(train_dataset, shuffle=True, seed=args.seed, drop_last=True),
    )
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // args.gradient_accumulation_steps // world_size,
        shuffle=False,
        num_workers=1,
        sampler=DistributedSampler(val_dataset, shuffle=False, seed=args.seed, drop_last=True),
    )
    print(f"Finished loading datasets, training samples: {len(train_dataset)}, validation samples: {len(val_dataset)} on rank {rank}")

    device = torch.device(rank)
    # Set the current device to make FSDP work
    torch.cuda.set_device(device)

    # Initialize the model on the current device
    if "llama" in args.model_type:
        model = LlamaForCausalLM.from_pretrained(args.model_type, device_map=device, torch_dtype=torch.float32, use_flash_attention_2=args.flash_attn)
    elif "mistral" in args.model_type or "zephyr" in args.model_type:
        model = MistralForCausalLM.from_pretrained(args.model_type, device_map=device, torch_dtype=torch.float32, use_flash_attention_2=args.flash_attn)
    elif "Qwen2" in args.model_type:
        model = Qwen2ForCausalLM.from_pretrained(args.model_type, device_map=device, torch_dtype=torch.float32, use_flash_attention_2=args.flash_attn)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    # We would like to be a bit conservative, and only cast the params, activations, and gradients to bfloat16.
    # Gradients will be reduced in float32, and the buffers (i.e. norms) will be stored in float32.
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)

    # We use ZeRO-3 to reduce the memory usage. ZeRO-2 is feasible but slower due to excessive memory re-allocation.
    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, auto_wrap_policy=wrap_policy, mixed_precision=mixed_precision)
    print(f"Finished initializing FSDP model on rank {rank}")

    optimizer = optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, fused=True)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=int(args.cycle_epochs * len(train_dataloader)),
        cycle_mult=1.0,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=int(args.warmup_epochs * len(train_dataloader)),
        gamma=args.lr_gamma,
    )
    print(f"Initialized optimizer and LR scheduler on rank {rank}")

    batch_loss = 0.0
    step_count = 0
    batch_count = 0
    last_batch_time = time.time()

    for epoch in range(args.num_epochs):
        # Set the epoch for the sampler
        train_dataloader.sampler.set_epoch(epoch)

        model.train()
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

        for step_count, batch in enumerate(train_dataloader):
            inputs = batch["input_ids"]
            labels = batch["labels"]
            mask = batch["attention_mask"]

            outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss /= args.gradient_accumulation_steps

            loss.backward()

            reduced_loss = loss.detach()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
            batch_loss += float(reduced_loss.item())

            if (step_count + 1) % args.gradient_accumulation_steps == 0:
                # Perform gradient clipping and optimizer step
                model.clip_grad_norm_(args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step()
                batch_count += 1

                if rank == 0:
                    print(f"Epoch {epoch}, Batch {batch_count}, Loss {batch_loss}, samples/sec {args.batch_size / (time.time() - last_batch_time)}")
                    log_writer.add_scalar("train_loss", batch_loss, batch_count)

                batch_loss = 0.0
                last_batch_time = time.time()

        model.eval()

        dist.barrier()

        eval_loss = 0
        eval_start_time = time.time()
        with torch.no_grad():
            for batch in valid_dataloader:
                inputs = batch["input_ids"]
                labels = batch["labels"]
                mask = batch["attention_mask"]

                outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)

                reduced_loss = outputs.loss.detach()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
                eval_loss += float(reduced_loss.item())

        dist.barrier()
        if rank == 0:
            eval_loss /= len(valid_dataloader)
            log_writer.add_scalar("eval_loss", eval_loss, batch_count)
            print(
                f"Batch count {batch_count}, Eval loss {eval_loss}, "
                + f"samples/sec {len(valid_dataloader) * args.gradient_accumulation_steps * world_size / (time.time() - eval_start_time)}"
            )

        save_model(model, rank, args.saved_model_path, "{}_latest".format(args.model_name))

    if rank == 0:
        log_writer.close()

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune models on natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./configs/llama3_steerable_opinion_qa_star.json", help="Path to config file")
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        args = DotMap(json.load(f))

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)