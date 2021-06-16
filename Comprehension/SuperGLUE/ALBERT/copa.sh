export TASK_NAME=copa
export MODEL_NAME=albert-xlarge-v2
# or 
# export MODEL_NAME=albert-xxlarge-v2

EXP_DIR=/your-export-path
export DATA_D=/your-data-path
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/your-path/jiant:$PYTHONPATH
export EXP_P=/your-path/${TASK_NAME}/${MODEL_NAME}


nohup python jiant/proj/simple/runscript.py \
    run_with_continue \
    --run_name copa-albert \
    --exp_dir ${EXP_P}/ \
    --data_dir ${DATA_D}/tasks \
    --hf_pretrained_model_name_or_path $MODEL_NAME \
    --tasks ${TASK_NAME} \
    --train_batch_size 32 \
    --num_train_epochs 20 \
    --max_seq_length 256 \
    --learning_rate 1e-5 \
    --seed 2021 \
    --eval_every_steps 13 \
    --save_checkpoint_every_steps 5000 \
    --do_save_best \
    --write_test_preds \
> ./nohup/$TASK_NAME-$MODEL_NAME.log 2>&1 &