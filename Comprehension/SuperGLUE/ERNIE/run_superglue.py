import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset

from knowledge_bert import BertTokenizer, BertForSequenceClassification
from utils import get_ent_id, get_ent_map, get_ents, merge_and_truncate

import transformers
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from metric.my_metrics import MyMetrics


task_to_sentence_keys = {
    "cb": ("premise", "hypothesis"),
    "wic": ("sentence1", "sentence2"),
    "boolq": ("passage", "question"),
    "wsc": ("text", "span1_text"),
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


def main():
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, training_args = parser.parse_args_into_dataclasses()

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

    # load model for finetuning
    model, _ = BertForSequenceClassification.from_pretrained('./ernie_base', num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('./ernie_base')
    
    logger.info("Loading entity embeddings ......")
    model.bert.set_ent_embeddings("./kg_embed/entity2vec.vec")
    logger.info(f"Loading finished, Embedding shape: {model.bert.ent_embeddings.weight.size()}")

    # load kg embed information for ernie
    ent_map, ent2id = get_ent_map(), get_ent_id()

    # convert datasets to examples which can be fed into plms
    if data_args.max_seq_length > tokenizer.max_len:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.max_len}). Using max_seq_length={tokenizer.max_len}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.max_len)

    sentence1_key, sentence2_key = task_to_sentence_keys[data_args.task_name]

    def preprocess_function(examples):
        sentence1 = examples[sentence1_key]
        if data_args.task_name == "wic":
            sentence1 = examples["word"] + ": " + sentence1
        if data_args.task_name == "wsc":
            words = sentence1.split()
            words[examples["span2_index"]] = "*" + words[examples["span2_index"]] + "*"
            sentence1 = ' '.join(words)
        tokens_a, entities_a = tokenizer.tokenize(sentence1, get_ents(sentence1, ent_map))
        # truncate
        if sentence2_key:
            sentence2 = examples[sentence2_key]
            tokens_b, entities_b = tokenizer.tokenize(sentence2, get_ents(sentence2, ent_map))
        else:
            tokens_b, entities_b = None, None
        tokens, ents, segment_ids = merge_and_truncate(tokens_a, tokens_b, entities_a, entities_b, max_seq_length)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        input_ent, ent_mask = [], []
        for ent in ents:
            if ent != "UNK" and ent in ent2id:
                input_ent.append(ent2id[ent])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_ = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        input_ent += padding_
        ent_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_ent) == max_seq_length
        assert len(ent_mask) == max_seq_length

        result = {"label": examples["label"], "input_ids": input_ids, "input_mask": input_mask,
                  "segment_ids": segment_ids, "input_ent": input_ent, "ent_mask": ent_mask}

        return result

    datasets = datasets.map(preprocess_function, batched=False)
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
