# Comprehension Ability

## GLUE

### Dataset

We get the datasets from [GLUE Benchmark](https://gluebenchmark.com/tasks). All datasets have been originally splited except MRPC. There are only training set and test set in MRPC dataset when we downloaded from the website mentioned above, thus we divided the training set into two parts: training set (3668) and development set (408).

### Checkpoint

We get the checkpoint of pretrained model from [Hugging Face](https://huggingface.co/models).

For ERNIE, we get the checkpoint provided by [thunlp/ERNIE](https://github.com/thunlp/ERNIE#pre-trained-model). You can download the pretrained language model and knowledge embedding through the method provided in this [link](https://github.com/thunlp/ERNIE#pre-trained-model), then place the downloaded <u>ernie_base</u> and <u>kg_embed</u> in this directory (e.g <u>./GLUE/ERNIE/</u>). And ERNIE needs to extract entity mentions in the sentences and link them to their corresponding entities in KGs. We use [TAGME](https://tagme.d4science.org/tagme/) to extract the entities, and you can find how to use TAGME [here](https://pypi.org/project/tagme/). (Note that the entity extraction process may take a lot of time.)

### Scripts

You can find the test scripts of each models in folders (named by the models' name) in this repository. In each folder, the document that named after model's name and size is the script to fine-tune and test corresponding model. For example, the scripts to fine-tune and test ALBERT are stored in <u>./GLUE/ALBERT/albert_xlarge.txt</u> and <u>./GLUE/ALBERT/albert_xxlarge.txt</u>.

Because fairseq don't provide function to predict results of test set, we write scripts to predict and these scripts are stored in the <u>./GLUE/ModelName/predict_script</u>.

For few-shot, we save the few-shot datasets and scripts to transform original dataset and make prediction on development datasets in the folder named <u>few-shot</u>.

### Test

We used [Transformers](https://github.com/huggingface/transformers), [jiant](https://github.com/nyu-mll/jiant) and [fairseq](https://github.com/pytorch/fairseq) to test models. If you want to use the frame mentioned above, you have to clone the repository and install it according to the guidance from GitHub.

GPT-2, ProphetNet, UniLM and XLNet are mainly fine-tuned and tested by Transformers. BERT and ALBERT are mainly fine-tuned and tested by jiant.  BART and RoBERTa are mainly fine-tuned by fairseq.  For ERNIE, we use and modify the official model implementation from [thunlp/ERNIE](https://github.com/thunlp/ERNIE#pre-trained-model).

In some tasks, we use different frames to get the best results. And some frames don't support especial tasks, so we use different frames to test these tasks.

You should change the path to datasets and models' checkpoints if you want to fine-tune and test model by using our scripts.

+ **fairseq**

  First of all, we prepare data by running the following script. <u>glue_data</u> means the dataset that you download from GLUE Benchmark. And `<glue_task_name>` is the name of this task. After running this script, it will create a folder named <u>TaskName_bin</u> under the current folder which is the prepared data. Script <u>preprocess_GLUE_tasks.sh</u> can be found in [fairseq repository](https://github.com/pytorch/fairseq).
  
    ```bash
  ./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>
    ```
  
  Here is an example to show how to use fairseq to test models. You can easily type it into terminal and it will fine-tune the model. The other test scripts can be found in folders of model's name.
  
  ```bash
  CUDA_VISIBLE_DEVICES=3 fairseq-train ../fairseq-master/QQP-bin/ \
      --restore-file ./bart_base/model.pt \
      --batch-size 32 \
      --max-tokens 4400 \
      --task sentence_prediction \
      --add-prev-output-tokens \
      --layernorm-embedding \
      --share-all-embeddings \
      --share-decoder-input-output-embed \
      --reset-optimizer --reset-dataloader --reset-meters \
      --required-batch-size-multiple 1 \
      --init-token 0 \
      --arch bart_base \
      --criterion sentence_prediction \
      --num-classes 2 \
      --dropout 0.1 --attention-dropout 0.1 \
      --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
      --clip-norm 0.0 \
      --lr-scheduler polynomial_decay --lr 1e-5 --total-num-update 113272 --warmup-updates 6796 \
      --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
      --max-epoch 10 \
      --find-unused-parameters \
      --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric
  ```
  
  Fairseq will save the best checkpoint of one tasks. So we should use the scripts to make prediction for test set. The predicting scripts can be found in folders of model's name.
  
  The hyper-parameters are copied from the GitHub and we changed some of them to get the best result.


+ **Transformers**

  We can use the script below to fine-tune models by using transformers. Transformers will downloaded dataset from website automatically, so we don't need to set the path to datasets. Script <u>run_glue.py</u> can be found in [transformers repository](https://github.com/huggingface/transformers).

  ```bash
  CUDA_VISIBLE_DEVICES=6 python ./examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path ../pretrained_model/unilm_base \
    --task_name mrpc \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --overwrite_output_dir \
    --output_dir ../result/unilm_base/mrpc
  ```

  For ProphetNet, because it doesn't support text classification tasks in transformers, we write a text-classification for it. We change the <u>run_glue.py</u> in Transformers and change `AutoModelForSequenceClassification` to our header. After that, we can test ProphetNet as other models.

+ **jiant**

  Jiant can fine-tune models by using the script below. You can change the model path to run different model and change the dataset path to fine-tune on different dataset.  Script <u>runscript.py</u> can be found in [jiant repository](https://github.com/nyu-mll/jiant).
  
  ```bash
  python jiant/proj/simple/runscript.py \
      run \
      --run_name albert_xlarge_wnli \
      --exp_dir ../gluedataset/ \
      --data_dir ../gluedataset/tasks \
      --hf_pretrained_model_name_or_path ../pretrained_model/albert_xlarge \
      --tasks wnli \
      --train_batch_size 32 \
      --num_train_epochs 5 \
      --seed=2021 \
      --do_save \
      --write_test_preds
  ```
  
  After fine-tuning, we use the script below to create prediction file.
  
  ```bash
  python ./jiant/scripts/benchmarks/benchmark_submission_formatter.py \
      --benchmark GLUE \
      --input_base_path ../test_preds/albert_xlarge/ \
      --tasks wnli mnli qqp qnli rte sst stsb cola mrpc\
      --output_path ../result/albert_xlarge/
  ```

+ **ERNIE**

  For ERNIE, use the script in <u>ernie.txt</u> directly for fine-tuning and prediction. For example, we use the following script to run on `STSB` dataset.

  ```bash
  python run_glue.py \
    --task_name stsb \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 128 \
    --learning_rate 1e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --num_train_epochs 10 \
    --output_dir ../results/ernie-thu/stsb/ \
    --warmup_ratio 0.0 \
    --seed 2021 \
    --load_best_model_at_end True \
    --metric_for_best_model combined_score \
    --evaluation_strategy epoch \
    --logging_steps 20 \
    --do_train \
    --do_eval
  ```

We doesn't test some models on especial tasks, because the results for those tasks can be found from papers.

Some models require to change some details in script, we give the changes in models' folder.

### Few-Shot

The datasets for few-shot was transformed from datasets on website. For training datasets, we chose some data from original datasets and create json file to save the data. We keep the datasets of same size are different from each other. You can find the scripts in <u>./GLUE/few-shot/change-CoLA.py</u> and <u>./GLUE/few-shot/change-QNLI.py</u>. For development and test datasets, we only change them to "json" file and didn't change the corpus.

We used Transformers to fine-tune models on few-shot. If you want to fine-tune other models, you have to change the dataset and the checkpoint of model. Script <u>run_glue.py</u> can be found in [transformers repository](https://github.com/huggingface/transformers).

```bash
CUDA_VISIBLE_DEVICES=1 python ./examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path ../pretrained_model/gpt2_medium \
  --train_file ../GlueDataset/QNLI-few-shot/train-50-0.json \
  --validation_file ../GlueDataset/QNLI-few-shot/dev.json \
  --do_train \
  --do_eval \
  --seed 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-5 \
  --overwrite_output_dir \
  --output_dir ../few-shot/gpt2-50-0/qnli
```

After fine-tuning model, you can use the scripts under folder <u>./GLUE/few-shot</u> to make prediction. You have to change the path to checkpoint and dataset. It suggests to use "tsv" file.

## SuperGLUE

### Dataset

We get the datasets from [SuperGLUE Benchmark](https://super.gluebenchmark.com/tasks). 

| Name                                 | Identifier |
| ------------------------------------ | ---------- |
| CommitmentBank                       | CB         |
| Choice of Plausible Alternatives     | COPA       |
| Multi-Sentence Reading Comprehension | MultiRC    |
| Recognizing Textual Entailment       | RTE        |
| Words in Context                     | WiC        |
| The Winograd Schema Challenge        | WSC        |
| BoolQ                                | BoolQ      |

### Scripts

For all classification tasks, we use the same approach to fine-tune all pretrained language models on the same task. For example, in order to run tests on `CommitmentBank` dataset, we need to add the following keys to <u>./SuperGLUE/fine_tune.py</u>：

```python
task_to_sentence_keys = {
    "cb": ("premise", "hypothesis"),
}
```

Then, for instance, we run the following code to fine-tune BERT-Large and run tests on `CommitmentBank` dataset：

```shell
nohup python finetune.py \
  --model_name_or_path bert-large-cased \
  --task_name cb \
  --max_seq_length 256 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --num_train_epochs 10 \
  --output_dir /$YOURPATH/ \
  --warmup_ratio 0.06 \
  --seed 2021 \
  --load_best_model_at_end True \
  --metric_for_best_model combined_score \
  --evaluation_strategy epoch \
  --logging_steps 4 \
> ./$YOURPATH/$TASK_NAME-$MODEL_NAME.log 2>&1 &
```

For other classification tasks, just add the correct keys to <u>./SuperGLUE/fine_tune.py</u> and modify the `--task_name` parameter in the shell script.

For non-classification tasks, we use similar approach to fine-tune and test, take <u>./SuperGLUE/finetune_multirc.py</u> as an example. 

In order to validate our test scripts, we also used jiant to do the same tests, and the results were consistent. For example, <u>./SuperGLUE/RoBERTa/wsc.sh</u> runs RoBERTa model on The Winograd Schema Challenge with jiant framework.
