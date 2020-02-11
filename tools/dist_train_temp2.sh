#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 7008\
    $(dirname "$0")/train_temp2.py $CONFIG --launcher pytorch ${@:3} --validate 1
