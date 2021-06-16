export CUDA_VISIBLE_DEVICES=1
export MODEL_NAME=gpt2-medium
export SUBSET_NAME=google_re

python main.py \
  --model_name_or_path $MODEL_NAME \
  --subset_name $SUBSET_NAME \
  --train_batch_size 8 \
  --eval_batch_size 16 \
  --evaluate_only False
