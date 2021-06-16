import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import is_main_process

from metric.my_metrics import MyMetrics


task_to_sentence_keys = {
    "cb": ("premise", "hypothesis"),
}


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: str = field(
        default=None,
        metadata={
            "help": "The name of the task to train on."
        }
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default="./plms/",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load super glue dataset
    if data_args.task_name in task_to_sentence_keys.keys():
        logger.info(f"Loading dataset {data_args.task_name}")
        datasets = load_dataset("super_glue.py", data_args.task_name)
        datasets.pop("test")  # pop test set and we will evaluate model on dev dataset
    else:
        raise KeyError(f"You should specify a task from {task_to_sentence_keys.keys()} for finetuning your model.")

    label_list = datasets["train"].features["label"].names
    num_labels = len(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}

    # load model for finetuning
    if model_args.model_name_or_path:
        # official unilm config.json don't have a required model_type key
        config_path = model_args.model_name_or_path
        if "unilm" in config_path:
            version = config_path.split("/")[-1]
            config_path = f"./config/{version}.json"
        config = AutoConfig.from_pretrained(
            config_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise ValueError("You should specify a model name or path for loading your model.")

    # add new pad token for some models like gpt2 which do not have pad token
    if "pad_token" not in tokenizer.special_tokens_map.keys():
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_tokenizer_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    # convert datasets to examples which can be fed into plms
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    sentence1_key, sentence2_key = task_to_sentence_keys[data_args.task_name]

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs
        # if label_to_id is not None and "label" in examples:
        #     result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        result["label"] = examples["label"]
        return result

    datasets = datasets.map(preprocess_function, batched=True)
    train_dataset = datasets["train"]
    valid_dataset = datasets["validation"]

    metric = MyMetrics(data_args.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = preds.argmax(-1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        # used to select best model
        result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Initialize our Trainer and finetune model on train dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-4)],
    )

    trainer.train()
    trainer.save_model()

    eval_result = trainer.evaluate(eval_dataset=valid_dataset)
    logger.info("*** Evaluate ***")

    output_eval_file = os.path.join(training_args.output_dir, f"eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")


if __name__ == '__main__':
    main()
