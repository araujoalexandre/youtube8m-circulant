#!/bin/bash

# config file start with the data type on the first line [video | frame]
# the exec args can be parse from the second line :
# --args1 value1 \
# --args2 value2 ... 
CONFIG_FILE="$CONFIG_DIR/train.cfg"
DATA_TYPE=$(head $CONFIG_FILE --lines 1)
PYTHON_ARGS=$(tail $CONFIG_FILE -n +2 | sed -e ':a' -e 'N' -e '$!ba' -e 's/\\\n/ /g')

DATE_STR=$(date '+%Y-%m-%d_%H.%M.%S')

export TRAIN_DIR="$MODEL_DIR/${DATA_TYPE}/${DATE_STR}"
export LOGS_DIR="$MODEL_DIR/${DATA_TYPE}/${DATE_STR}_logs"

mkdir -p ${TRAIN_DIR}
mkdir -p ${LOGS_DIR}

# copy config file to train folder
cp ${CONFIG_FILE} ${LOGS_DIR}/

export DATA_DIR="${DATA_PREFIX_DIR}/${DATA_TYPE}"

CMD="python3 $EXEC_DIR/train.py ${PYTHON_ARGS} --num_gpu ${NUM_GPU} &>> $LOGS_DIR/out.log"
echo "About to execute $CMD"
eval $CMD

