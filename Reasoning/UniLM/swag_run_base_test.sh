# run fine-tuning
DATA_DIR=./data/swag
OUTPUT_DIR=swag-output
MODEL_RECOVER_PATH=unilm1-base-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=./BERT/bert-base-cased
export CUDA_VISIBLE_DEVICES=0
python src/run_swag.py  --do_predict \
  --bert_model bert-base-cased \
  --data_dir ${DATA_DIR} \
  --pred_epoch 4 \
  --output_dir ${OUTPUT_DIR}/unilm-base-output \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 64  \
  --eval_batch_size 64 