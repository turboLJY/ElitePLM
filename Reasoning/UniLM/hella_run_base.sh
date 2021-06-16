DATA_DIR=./data/hellaswag
OUTPUT_DIR=hellaswag-output
MODEL_RECOVER_PATH=unilm1-base-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=./BERT/bert-base-cased
export CUDA_VISIBLE_DEVICES=0

python src/run_hellaswag.py  --do_train --do_eval \
  --bert_model bert-base-cased \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR}/unilm-base-output \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 64  \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --warmup_proportion 0.1  \
  --num_train_epochs 10  