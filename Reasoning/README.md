# Reasoning Ability

## Dataset

We get each dataset from the public website attached to its original paper.
- [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa)
- [ROCStories](https://cs.rochester.edu/nlp/rocstories/) (validation set winter 2016; test set winter 2018)
- [SWAG](https://rowanzellers.com/swag/)
- [HellaSwag](https://rowanzellers.com/hellaswag/)
- [Sense Making](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation)
- [ARCT](https://github.com/UKPLab/argument-reasoning-comprehension-task)

We take the official split of all these datasets except ROCStories. Following [exsiting works](https://arxiv.org/pdf/1905.07504.pdf) of ROCStories, we use original validation set for training, and split it into training set (80%) and validation set (20%), then we use original test set for testing, you can get them under <u>data/ROCStories</u> directory.

## Checkpoint

We get the checkpoint of pretrained model from [Hugging Face](https://huggingface.co/models).

For ERNIE, we get the checkpoint provided by [thunlp/ERNIE](https://github.com/thunlp/ERNIE#pre-trained-model).

## Scripts

You can find the test scripts of each models in folders (named by models' name) in this repository. Because all these tasks are multiple-choice tasks, the scripts are modified mainly based on Hugging Face multiple-choice [examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/multiple-choice) and fairseq [examples](https://github.com/pytorch/fairseq/tree/master/examples/roberta/commonsense_qa).

- ALBERT/ RoBERTa/ GPT-2/ XLNet / ProphetNet/ T5/ ERNIE: the script with model name is the command in terminal, like <u>ALBERT.txt</u> for ALBERT model. The file with dataset name is the script to fine-tune for corresponding dataset, like <u>run_coqa.py</u> for CommonsenseQA. The result of CommonsenseQA for T5 can be found in its [original paper](https://arxiv.org/pdf/1910.10683.pdf).
- BART: In <u>./BART</u>, the shell scripts are commands for training and the <u>test</u> directory contains scripts for testing. The <u>task</u> directory contains fairseq tasks of each dataset, and for CommonsenseQA, there has already been a task script in fairseq. 
- UniLM: In <u>./UniLM</u>, the shell scripts are commands for training and testing, and the <u>src</u> directory contains scripts for each dataset.
- BERT: The result of these datasets for BERT can be found in its [original paper](https://www.aclweb.org/anthology/N19-1423.pdf).

### Test

We use the following frameworks in experiments. If you want to use the framework mentioned above, you have to clone the repository and install it according to the guidance from GitHub.
  - ALBERT/ RoBERTa/ GPT-2/ XLNet/ ProphetNet/ T5: we use [Hugging Face](https://github.com/huggingface/transformers). In particular, T5 is tested by `AutoModelForSeq2SeqLM` module and others are tested by `AutoModelForMultipleChoice` module. For GPT-2 and ProphetNet, we add <u>GPT-2_Model.py</u> and <u>ProphetNet_Model.py</u> for each and replace `AutoModelForMultipleChoice` module, because there isn't any realization of multiple-choice task.
  - BART: we use [fairseq](https://github.com/pytorch/fairseq) to test models.
  - UniLM: we modify the [code](https://github.com/microsoft/unilm/tree/master/unilm-v1) of UniLM paper to test models.
  - ERNIE: we modify the [code](https://github.com/thunlp/ERNIE) of ERNIE paper to test models.

You should change the path to datasets and models' checkpoints if you want to fine-tune and test model by using our scripts.

+ **ALBERT/RoBERTa/GPT-2/ XLNet/ ProphetNet/T5**
  
    Here is an example to show how to use transformers to test ALBERT-XL on SWAG dataset. After downloading the dataset, you can easily type the command into terminal and fine-tune the model. The other test scripts can be found in folders (named by model's name).
  
    ```bash
    python run_swag.py \
    --model_name_or_path albert-xlarge-v2 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 6129 \
    --save_strategy steps \
    --save_steps 6129 \
    --learning_rate 1e-6 \
    --num_train_epochs 5 \
    --output_dir swag_albert_xl_output \
    --per_gpu_eval_batch_size=4 \
    --per_device_train_batch_size=4 \
    --overwrite_output \
    --train_file data/swag/train.csv \
    --validation_file data/swag/val.csv \
    --cache_dir  albert_xlarge
    ```
    To get the predictions of test set, you can run following command and the results could be found in <u>test_results</u> repository. 
    ```bash  
    python run_swag.py \
    --model_name_or_path swag_albert_xl_output \
    --do_predict \
    --per_gpu_eval_batch_size=32 \
    --test_file data/swag/test.csv \
    ```
    RoBERTa/GPT-2/XLNet/ProphetNet/T5 can be tested similarly according to commands for each.
    
+ **BART**

    Here is an example to show how to use faiseq to test BART-Large model on SWAG dataset. 
  ```bash
   ./bart_swag_run_large
  ```
   To get the predictions of test set, you can run following command and the results could be found in <u>test_results</u> repository. 
    ```bash  
    python test/test-swag.py
    ```
  
+ **UniLM**

    Here is an example to show how to test UniLM-Large model on SWAG dataset. 
  ```bash
   ./swag_run_large
  ```
   To get the predictions of test set, you can run following command and the results could be found in <u>test_results</u> repository. 
    ```bash  
    ./swag_run_large_test
    ```
+ **ERNIE**

    Here is an example to show how to test ERNIE model on SWAG dataset, and the predictions for test set could be found in <u>test_results</u> repository. 
  ```bash
   python swag_ernie.py \
   --train_file ./data/swag/train.csv \
   --validation_file ./data/swag/val.csv \
   --test_file ./data/swag/test.csv \
   --max_seq_length 128 \
   --per_device_train_batch_size 12 \
   --per_device_eval_batch_size 24 \
   --learning_rate 1e-5 \
   --num_train_epochs 10 \
   --output_dir results/swag \
   --load_best_model_at_end True \
   --metric_for_best_model combined_score \
   --evaluation_strategy epoch \
   --logging_steps 10
  ```


We doesn't test some models on especial tasks, because the results for those tasks can be found from papers.
