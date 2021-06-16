# Memory Ability

## Dataset

### LAMA

LAMA refers to the datasets used in paper "Language Models as Knowledge Bases?" for analyzing the factual and commonsense knowledge contained in pretrained language models. It contains four sub-datasets and we use datasets library provided by Hugging Face for both downloading and preprocessing them. For example, to get the G-RE sub-dataset (you need to first `pip install datasets`):

```python
from datasets import load_dataset
dataset = load_dataset("lama.py", "google_re")
```

or you can get the original dataset files at https://dl.fbaipublicfiles.com/LAMA/data.zip.

### Wikipedia

For Wikipedia, we download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract a small part of the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then randomly sample 100,000 sentences. We wrap it into a Dataset class which can be loaded with:

```python
from datasets import Dataset
dataset = Dataset.load_from_disk("./wiki")
```

## Checkpoint

We get the checkpoints of pretrained language models from [Hugging Face](https://huggingface.co/models). And we construct a connector class for each pretrained language model to use them as the same as [LAMA](https://github.com/facebookresearch/LAMA#lama-language-model-analysis).

## Scripts

You can simply change the name of dataset and model's checkpoint to run experiments with the scripts below.

- Zero-Shot Test

```bash
python main.py \
  --model_name_or_path bert-large-cased \
  --subset_name google_re \
  --eval_batch_size 16
```

- Efficiency Test

For efficiency test, we train model 3 epochs and evaluate it's performance 50 times during training. The evaluate results will be saved in <u>output_dir</u>. If you want to change the number of epochs or evaluate times , please edit the train method of particular model's connector.

```bash
python main.py \
  --model_name_or_path bert-large-cased \
  --subset_name google_re \
  --train_batch_size 8 \
  --eval_batch_size 16 \
  --evaluate_only False \
  --output_dir ./output
```
