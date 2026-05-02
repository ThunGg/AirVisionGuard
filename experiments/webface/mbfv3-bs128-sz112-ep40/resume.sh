#!/bin/bash
work_path=$(dirname $0)
python -u main.py \
    --config $work_path/config.yaml \
    --resume \
    --load-path $work_path/checkpoints/ckpt_epoch_$1.pth.tar
