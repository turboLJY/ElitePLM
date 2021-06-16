MAX_UPDATES=1570 	  # Number of training steps.
WARMUP_UPDATES=100    # Linearly increase LR over this many steps.
LR=1e-05              # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=64      # Batch size.
SEED=1                # Random seed.
BART_PATH=bart.base/model.pt
DATA_DIR=./data/sen-random
CKPT_DIR=sen2_checkpoints_base
#first_train: --restore-file $BART_PATH

# we use the --user-dir option to load the task 
FAIRSEQ_PATH=fairseq
FAIRSEQ_USER_DIR=task/fairseq_sen2

CUDA_VISIBLE_DEVICES=0 fairseq-train --ddp-backend=legacy_ddp \
    $DATA_DIR \
    --user-dir $FAIRSEQ_USER_DIR \
    --restore-file $BART_PATH  \
    --save-dir $CKPT_DIR \
    --reset-optimizer --reset-dataloader --reset-meters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task sen2 --init-token 0 --bpe gpt2 \
    --arch bart_base  \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_ranking --num-classes 3 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06  \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
    --batch-size $MAX_SENTENCES \
    --max-update $MAX_UPDATES \
    --log-format simple --log-interval 25 \
    --seed $SEED \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --clip-norm 0.1 \
    --find-unused-parameters \
    --no-epoch-checkpoints 
