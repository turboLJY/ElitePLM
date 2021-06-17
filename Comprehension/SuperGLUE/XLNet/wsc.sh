export HF_DATASETS_CACHE="./cache/"
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=wsc
export MODEL_NAME=xlnet-base-cased

nohup python finetune.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 1e-5 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --num_train_epochs 15 \
  --output_dir ./results/$TASK_NAME/$MODEL_NAME/ \
  --warmup_ratio 0.06 \
  --seed 2021 \
  --load_best_model_at_end True \
  --metric_for_best_model combined_score \
  --evaluation_strategy epoch \
  --logging_steps 20 \
> ./nohup/$TASK_NAME-$MODEL_NAME.log 2>&1 &


export HF_DATASETS_CACHE="./cache/"
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=wsc
export MODEL_NAME=xlnet-large-cased

nohup python finetune.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-5 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --num_train_epochs 15 \
  --output_dir ./results/$TASK_NAME/$MODEL_NAME/ \
  --warmup_ratio 0.06 \
  --seed 2021 \
  --load_best_model_at_end True \
  --metric_for_best_model combined_score \
  --evaluation_strategy epoch \
  --logging_steps 20 \
> ./nohup/$TASK_NAME-$MODEL_NAME.log 2>&1 &