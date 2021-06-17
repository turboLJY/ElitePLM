export HF_DATASETS_CACHE="./cache/"
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=multirc
export MODEL_NAME=xlnet-base-cased

nohup python finetune_multirc.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --max_seq_length 448 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --output_dir ./results/$TASK_NAME/$MODEL_NAME/ \
  --warmup_ratio 0.06 \
  --seed 2021 \
  --logging_steps 100 \
  --gradient_accumulation_steps 4 \
  --eval_times 10 \
  --metric_for_best_model exact_match \
> ./nohup/$TASK_NAME-$MODEL_NAME.log 2>&1 &


export HF_DATASETS_CACHE="./cache/"
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=multirc
export MODEL_NAME=xlnet-large-cased

nohup python finetune_multirc.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --max_seq_length 448 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --output_dir ./results/$TASK_NAME/$MODEL_NAME/ \
  --warmup_ratio 0.06 \
  --seed 2021 \
  --logging_steps 100 \
  --gradient_accumulation_steps 4 \
  --eval_times 10 \
  --metric_for_best_model exact_match \
> ./nohup/$TASK_NAME-$MODEL_NAME.log 2>&1 &