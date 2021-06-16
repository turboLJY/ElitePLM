# Composition

## Dataset

- CNN / Daily Mail

We use the original data in [UniLM](https://github.com/microsoft/unilm), which can be downloaded from [the provided link of UniLM](https://drive.google.com/open?id=1jiDbDbAsqy_5BM79SmX6aSu5DQVCAZq1).

- GigaWord

We use the original data in [UniLM](https://github.com/microsoft/unilm), which can be downloaded from [the provided link of UniLM](https://drive.google.com/open?id=1USoQ8lJgN8kAWnUnRrupMGrPMLlDVqlV).

- SQuAD

We use the original data in [UniLM](https://github.com/microsoft/unilm), which can be downloaded from [the provided link of UniLM](https://drive.google.com/open?id=11E3Ij-ctbRUTIQjueresZpoVzLMPlVUZ).

- WritingPrompts

We use the data from [fairseq](https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/stories/README.md), which can be downloaded from [the provided link of fairseq](https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz). 

Considering the max sequence length of each pretrained language model, we drop the data of which the length is over the max length of any pretrained model. Indeed, we drop the data more than 510 tokens after tokenization. We provide <u>./process.py</u> to conduct drop.

## Checkpoint

We get the checkpoint of pretrained model from [Hugging Face](https://huggingface.co/models).

## Script

Due to the source code may break the anonymous rules, we only release the core code to implement GPT-2, UniLM, T5, BART and ProphetNet under the framework of Hugging Face in <u>MODEL_NAME.py</u>. We will release the whole script after being accepted.

