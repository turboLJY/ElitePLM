import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset

from knowledge_bert import BertTokenizer, BertForMultipleChoice
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


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
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
        data_args, training_args, left = parser.parse_args_into_dataclasses()
        print(left)

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

    # load dataset    
    if data_args.train_file is not None or data_args.validation_file is not None or data_args.test_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file    
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, cache_dir=training_args.output_dir)   

    def preprocess_function(examples):
        claim, reason, choices = examples["sent1"], examples["sent2"], (examples["ending0"], examples["ending1"], examples["ending2"], examples["ending3"])
        result = {"label": examples["label"], "input_ids": [], "attention_mask": [],
                  "token_type_ids": [], "input_ent": [], "ent_mask": []}

        for choice in choices:
            sentence = " " + claim + reason + " " + choice
            time_a = 0
            
            while time_a < 10:
                try:   
                    tokens_a, entities_a = tokenizer.tokenize(sentence, get_ents(sentence, ent_map))
                    break
                except:
                    time_a += 1
                    continue
            # truncate
            tokens_b, entities_b = None, None
            time_b = 0
            while time_b < 10:
                try:
                    tokens, ents, segment_ids = merge_and_truncate(tokens_a, tokens_b, entities_a, entities_b, max_seq_length)
                    break
                except:
                    time_b += 1
                    continue
                    
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

            result["input_ids"].append(input_ids)
            result["attention_mask"].append(input_mask)
            result["token_type_ids"].append(segment_ids)
            result["input_ent"].append(input_ent)
            result["ent_mask"].append(ent_mask)

        return result

    # this process will take some minutes for the tokenizer's version implemented by ernie is too old,
    # which don't support fast and batch processing
    datasets = datasets.map(preprocess_function, batched=False,num_proc=16)
    train_dataset = datasets["train"]
    valid_dataset = datasets["validation"]
    test_dataset = datasets["test"]
    train_dataset.save_to_disk("swag/train")
    valid_dataset.save_to_disk("swag/valid")
    test_dataset.save_to_disk("swag/test")

    # load model for finetuning
    model, _ = BertForMultipleChoice.from_pretrained('./ernie_base', num_choices=4) ##num_choices
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
    )

    trainer.train()
    trainer.save_model()

    logger.info("*** Predict ***")
    predictions, labels, metrics = trainer.predict(test_dataset)

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    prediction_ids = np.argmax(predictions, axis=1)
    output_test_file = os.path.join('test_results', "swag_ernie.csv")
    if trainer.is_world_process_zero():
        with open(output_test_file, "w") as f:
            writer = csv.writer(f)
            for index, item in enumerate(prediction_ids):
                if index<10:
                    print([index]+[item])
                writer.writerow([index]+[item])


if __name__ == '__main__':
    main()
