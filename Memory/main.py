import argparse
import os
import torch
from sklearn.metrics import accuracy_score
from utils import load_data
from connector.auto_connector import auto_connector


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Testing PLMs on LAMA dataset.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default=None,
        help="The name of the dataset to train or test on."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--evaluate_only",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output"
    )
    parser.add_argument()
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(f"Running model: {args.model_name_or_path} on dataset: {args.subset_name}.")

    config = args.model_name_or_path
    if "unilm" in config:
        version = config.split("/")[-1]
        config = f"./config/{version}.json"

    model_connector = auto_connector(
        model_name_or_path=args.model_name_or_path, cache_dir=args.cache_dir, config=config
    )
    dataset = load_data(args.subset_name, model_connector)

    if args.evaluate_only:
        predictions, references = model_connector.get_results(dataset=dataset,
                                                              batch_size=args.eval_batch_size,
                                                              device=device)
        print(accuracy_score(predictions, references))
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        filename = args.model_name_or_path.split("/")[-1] + "-" + args.subset_name + ".csv"
        results = open(os.path.join(args.output_dir, filename), "w")
        model_connector.train(dataset=dataset,
                              train_batch_size=args.train_batch_size,
                              eval_batch_size=args.eval_batch_size,
                              device=device,
                              results=results)
        results.close()

    print(f"Vocab size of {args.model_name_or_path}: {model_connector.tokenizer.vocab_size}.")


if __name__ == '__main__':
    main()
