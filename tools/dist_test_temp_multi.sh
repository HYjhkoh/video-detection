#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

CONFIG=$1
CHECKPOINT=$2
OUT=$3
GPUS=$4

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 7001 \
    $(dirname "$0")/test_temp.py $CONFIG $CHECKPOINT --out $OUT --launcher pytorch ${@:5}
