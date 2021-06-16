export TASK_NAME=wsc
export MODEL_NAME=bert-large-cased 
# or 
# export MODEL_NAME=bert-base-cased 

EXP_DIR=/your-export-path
export DATA_D=/your-data-path
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/your-path/jiant:$PYTHONPATH
export EXP_P=/your-path/${TASK_NAME}/${MODEL_NAME}


nohup python jiant/proj/simple/runscript.py \
    run_with_continue \
    --run_name simple \
    --exp_dir ${EXP_DIR}/ \
    --data_dir ${EXP_DIR}/tasks \
    --hf_pretrained_model_name_or_path $MODEL_NAME \
    --tasks ${TASK_NAME} \
    --train_batch_size 16 \
    --num_train_epochs 50 \
    --max_seq_length 128\
    --learning_rate 1e-5\
    --seed 2021 \
    --eval_every_steps 50 \
    --save_checkpoint_every_steps 1000 \
    --do_save_best \
    --write_test_preds \
> ./nohup/$TASK_NAME-$MODEL_NAME.log 2>&1 & 