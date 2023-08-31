#!/bin/bash

CONFIGS_NAME="car_auto_T5_rnn_del_focal_l1l2_train"  # Need to set
ACTIVATE_ENV="conda activate nvtf1.15"  # Need to set

TRAIN_DIR="./checkpoints/${CONFIGS_NAME}/"

# --dataset_split_file to set
PY_CMD="python train_v3_mod_graph_gen.py \
--dataset_split_file ./splits/train_car.txt \
configs/${CONFIGS_NAME}_train_config \
configs/${CONFIGS_NAME}_config"

mkdir ${TRAIN_DIR}

INTERVAL="4"
gnome-terminal --tab -t "record pidstat CPU" -- bash -c "pidstat ${INTERVAL} -l  2>&1 | tee -a ${TRAIN_DIR}训练中运行的进程CPU信息.txt"&
gnome-terminal --tab -t "record pidstat 内存" -- bash -c "pidstat ${INTERVAL} -r  2>&1 | tee -a ${TRAIN_DIR}训练中运行的进程内存信息.txt"&
gnome-terminal --tab -t "record pidstat 任务时间片切换" -- bash -c "pidstat ${INTERVAL} -w  2>&1 | tee -a ${TRAIN_DIR}训练中运行的进程任务时间片切换信息.txt"&
gnome-terminal --tab -t "record CPU" -- bash -c "mpstat -P ALL ${INTERVAL}  2>&1 | tee -a ${TRAIN_DIR}训练中CPU信息.txt"&
gnome-terminal --tab -t "record GPU" -- bash -c "nvidia-smi -l ${INTERVAL}  2>&1 | tee -a ${TRAIN_DIR}训练中GPU信息.txt"&

sleep 2s
gnome-terminal --tab --active -t "record run.py" -- bash -ic "
${ACTIVATE_ENV}
${PY_CMD} 2>&1 | tee -a ${TRAIN_DIR}训练打印信息.txt
exec bash"&