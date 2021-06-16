DATA_DIR=./data/sen
OUTPUT_DIR=sen1-output
MODEL_RECOVER_PATH=unilm1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=./BERT/bert-large-cased
export CUDA_VISIBLE_DEVICES=0
python src/run_sen1.py  --do_predict \
  --bert_model bert-large-cased \
  --data_dir ${DATA_DIR} \
  --pred_epoch 2 \
  --output_dir ${OUTPUT_DIR}/unilm-large-output \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 64  \
  --eval_batch_size 32 