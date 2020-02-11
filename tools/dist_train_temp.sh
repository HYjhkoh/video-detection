#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 7007\
    $(dirname "$0")/train_temp.py $CONFIG --launcher pytorch ${@:3} --validate 1