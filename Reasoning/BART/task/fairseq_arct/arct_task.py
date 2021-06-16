# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import csv

import numpy as np
import torch
from fairseq.data import (
    Dictionary,
    IdDataset,
    ListDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    data_utils,
    encoders,
)
from fairseq.tasks import LegacyFairseqTask, register_task


@register_task("arct")
class ARCTTask(LegacyFairseqTask):
    """Task to finetune BART for ARCT."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data", metavar="DIR", help="path to data directory; we load <split>.jsonl"
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument("--num-classes", type=int, default=2)

    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab
        self.mask = vocab.add_symbol("<mask>")

        self.bpe = encoders.build_bpe(args)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert (
            args.criterion == "sentence_ranking"
        ), "Must set --criterion=sentence_ranking"

        # load data and label dictionaries
        vocab = cls.load_dictionary(os.path.join(args.data, "dict.txt"))
        print("| dictionary: {} types".format(len(vocab)))

        return cls(args, vocab)

    def load_dataset(
        self, split, epoch=1, combine=False, data_path=None, return_only=False, **kwargs
    ):
        """Load a given dataset split. 这里更改数据加载方式

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def binarize(s, append_bos=False):
            if self.bpe is not None:
                s = self.bpe.encode(s)
            tokens = self.vocab.encode_line(
                s,
                append_eos=True,
                add_if_not_exist=False,
            ).long()
            if append_bos and self.args.init_token is not None:
                tokens = torch.cat([tokens.new([self.args.init_token]), tokens])
            return tokens

        if data_path is None:
            data_path = os.path.join(self.args.data, split + ".txt") #dataname/train/valid/test.csv
        if not os.path.exists(data_path):
            raise FileNotFoundError("Cannot find data: {}".format(data_path))

        src_tokens = [[] for i in range(self.args.num_classes)] #按类别分
        src_lengths = [[] for i in range(self.args.num_classes)]
        labels = []

        
        with open(data_path) as h:
            lines = list(csv.reader(h, delimiter="\t"))
            for id_, row in enumerate(lines[1:]):
                label = -1 if split == "test" else int(row[3])   
                #print(label)
                labels.append(label)
                question = row[4] + row[5]#startphrase
                #print(question)
                #assert len(example["endings"]) == self.args.num_classes
                # format: `<s> Q: Where would I not want a fox? </s> A: hen house </s>`
                question = "Q: " + question
                question_toks = binarize(question, append_bos=True)
                for i, choice in enumerate(row[1:3]):
                    src = "A: " + choice
                    #print(src)
                    src_bin = torch.cat([question_toks, binarize(src)])
                    src_tokens[i].append(src_bin)
                    src_lengths[i].append(len(src_bin))
                #print(len(src_tokens))
        print('-----------------------------------------------------')
        print([len(src_tokens[i]) for i in range(self.args.num_classes)])
        print('-----------------------------------------------------')
        assert all(len(src_tokens[0]) == len(src_tokens[i]) for i in range(self.args.num_classes))
        assert len(src_tokens[0]) == len(src_lengths[0])
        assert len(labels) == 0 or len(labels) == len(src_tokens[0])

        for i in range(self.args.num_classes):
            src_lengths[i] = np.array(src_lengths[i])
            src_tokens[i] = ListDataset(src_tokens[i], src_lengths[i])
            src_lengths[i] = ListDataset(src_lengths[i])

        dataset = {
            "id": IdDataset(),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens[0], reduce=True),
        }

        for i in range(self.args.num_classes):
            dataset.update(
                {
                    "net_input{}".format(i + 1): {
                        "src_tokens": RightPadDataset(
                            src_tokens[i],
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        "src_lengths": src_lengths[i],
                    }
                }
            )

        if len(labels) > 0:
            dataset.update({"target": RawLabelDataset(labels)})

        dataset = NestedDictionaryDataset(
            dataset,
            sizes=[np.maximum.reduce([src_token.sizes for src_token in src_tokens])],
        )

        with data_utils.numpy_seed(self.args.seed):
            dataset = SortDataset(
                dataset,
                # shuffle
                sort_order=[np.random.permutation(len(dataset))],
            )

        print("| Loaded {} with {} samples".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.register_classification_head(
            "sentence_classification_head",
            num_classes=1,
        )

        return model

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab
