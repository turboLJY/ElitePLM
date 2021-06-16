import argparse
import logging
import os
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import datasets as datasets_lib
from datasets import load_dataset, Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from sklearn.metrics import f1_score


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The task to be finetuned and evaluated."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./plms/",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Num of steps for logging model loss.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluating dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--metric_for_best_model", type=str, default=None, help="The metric to use to compare two different models"
    )
    parser.add_argument(
        "--eval_times",
        type=int,
        default=10,
        help="Num of total evaluation times during training."
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets_lib.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets_lib.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Set seed before initializing model.
    set_seed(args.seed)

    # load super glue dataset
    assert args.task_name == "multirc"
    logger.info(f"Loading dataset {args.task_name}")
    datasets = load_dataset("super_glue.py", args.task_name)
    datasets.pop("test")  # pop test set and we will evaluate model on dev dataset
    valid_idx = datasets["validation"]["idx"]  # store idx for evaluating

    label_list = datasets["train"].features["label"].names
    num_labels = len(label_list)

    # load model for finetuning
    if args.model_name_or_path:
        # official unilm config.json don't have a required model_type key
        config_path = args.model_name_or_path
        if "unilm" in config_path:
            version = config_path.split("/")[-1]
            config_path = f"./config/{version}.json"
        config = AutoConfig.from_pretrained(
            config_path,
            num_labels=num_labels,
            cache_dir=args.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            use_fast=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models'
        )
    else:
        raise ValueError("You should specify a model name or path for loading your model.")

    # add new pad token for some models like gpt2 which do not have pad token
    if "pad_token" not in tokenizer.special_tokens_map.keys():
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    # convert datasets to examples which can be fed into plms
    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    padding = "max_length"
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        sentence2 = []
        for question, answer in zip(examples["question"], examples["answer"]):
            sentence2.append(question + ' answer: ' + answer)
        args = (examples["paragraph"], sentence2)
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        result["label"] = examples["label"]
        return result

    datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)
    train_dataset = datasets["train"]
    valid_dataset = datasets["validation"]

    def evaluate_multirc(preds, labels, indices):
        """
        Computes F1 score and Exact Match for MultiRC predictions.
        """
        question_map = {}
        for pred, label, idx in zip(preds, labels, indices):
            question_id = "{}-{}".format(idx["paragraph"], idx["question"])
            if question_id in question_map:
                question_map[question_id].append((pred, label))
            else:
                question_map[question_id] = [(pred, label)]
        f1s, ems = [], []
        for question, preds_labels in question_map.items():
            question_preds, question_labels = zip(*preds_labels)
            f1 = f1_score(y_true=question_labels, y_pred=question_preds, average="macro")
            f1s.append(f1)
            em = int(sum([p == l for p, l in preds_labels]) == len(preds_labels))
            ems.append(em)
        f1_m = sum(f1s) / len(f1s)
        em = sum(ems) / len(ems)
        f1_a = f1_score(y_true=labels, y_pred=preds)
        return {"exact_match": em, "f1_m": f1_m, "f1_a": f1_a}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(valid_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.98))

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    eval_steps = max_train_steps // args.eval_times

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Num steps per evaluation = {eval_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_metric = {"exact_match": 0.0, "f1_m": 0.0, "f1_a": 0.0}
    major = args.metric_for_best_model

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps >= max_train_steps:
                    break

                if completed_steps % args.logging_steps == 0:
                    logger.info(f"Loss: {loss}")

                if completed_steps % eval_steps == 0:
                    logger.info(f"***** Running evaluating at steps {completed_steps} *****")
                    model.eval()
                    with torch.no_grad():
                        predictions, references = np.array([], dtype=int), np.array([], dtype=int)
                        for _, eval_batch in enumerate(eval_dataloader):
                            outputs = model(**eval_batch)
                            predicts = outputs.logits.argmax(dim=-1)
                            predictions = np.append(predictions, predicts.data.cpu().numpy())
                            references = np.append(references, eval_batch["labels"].data.cpu().numpy())
                        metric = evaluate_multirc(predictions, references, valid_idx)
                        for k, v in metric.items():
                            logger.info(f"{k}: {v}")
                        if metric[major] >= best_metric[major]:
                            best_metric = metric

    logger.info(f"***** Running evaluating at end *****")
    model.eval()
    with torch.no_grad():
        predictions, references = np.array([], dtype=int), np.array([], dtype=int)
        for _, eval_batch in enumerate(eval_dataloader):
            outputs = model(**eval_batch)
            predicts = outputs.logits.argmax(dim=-1)
            predictions = np.append(predictions, predicts.data.cpu().numpy())
            references = np.append(references, eval_batch["labels"].data.cpu().numpy())
        metric = evaluate_multirc(predictions, references, valid_idx)
        for k, v in metric.items():
            logger.info(f"{k}: {v}")
        if metric[major] >= best_metric[major]:
            best_metric = metric

    logger.info("***** Best Performance *****")
    for k, v in best_metric.items():
        logger.info(f"{k}: {v}")


if __name__ == '__main__':
    main()
