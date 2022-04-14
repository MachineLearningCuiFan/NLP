PRETRAINED_DIR="pretrained"
DATE_DIR="data"
OUTPUT_DIR="output"

python  src/run.py \
    --model_type bert \
    --model_name_or_path $PRETRAINED_DIR \
    --output_dir $OUTPUT_DIR  \
    --do_train --do_eval   \
    --data_dir $DATE_DIR \
    --train_file trainall.times2.pkl \
    --dev_file test.sighan13.pkl \
    --dev_label_file test.sighan13.lbl.tsv \
    --order_metric sent-detect-f1  \
    --metric_reverse  \
    --num_save_ckpts 5 \
    --remove_unused_ckpts  \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 100  \
    --learning_rate 5e-5 \
    --num_train_epochs 10  \
    --seed 17 \
    --warmup_steps 10000  \
    --eval_all_checkpoints \
    --overwrite_output_dir \
    --local_rank -1
