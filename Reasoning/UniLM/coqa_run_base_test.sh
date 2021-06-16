# run fine-tuning
DATA_DIR=./data/coqa
OUTPUT_DIR=coqa-output
MODEL_RECOVER_PATH=unilm1-base-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=./BERT/bert-base-cased
export CUDA_VISIBLE_DEVICES=0
python src/run_coqa.py  --do_predict \
  --bert_model bert-base-cased \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR}/unilm-base-output \
  --pred_epoch 20 \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 64  \
  --eval_batch_size 64 