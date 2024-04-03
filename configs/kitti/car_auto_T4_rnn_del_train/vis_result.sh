#!/bin/bash

CONFIGS_NAME="car_auto_T4_rnn_del_train"  # Need to set
ACTIVATE_ENV="conda activate tf1.15.0"  # Need to set

TRAIN_DIR="./checkpoints/${CONFIGS_NAME}/"

# --dataset_split_file to set
PY_CMD="python vis_result.py \
${TRAIN_DIR} \
--dataset_root_dir ../dataset/kitti \
--dataset_split_file ./splits/3DOP_splits/val.txt \
--output_dir ${TRAIN_DIR} \
--level 1"

mkdir ${TRAIN_DIR}

#gnome-terminal --tab -t "record pidstat CPU" -- bash -c "pidstat 1 -l  2>&1 | tee -a ${TRAIN_DIR}评估中运行的进程CPU信息.txt"&
#gnome-terminal --tab -t "record pidstat 内存" -- bash -c "pidstat 1 -r  2>&1 | tee -a ${TRAIN_DIR}评估中运行的进程内存信息.txt"&
#gnome-terminal --tab -t "record pidstat 任务时间片切换" -- bash -c "pidstat 1 -w  2>&1 | tee -a ${TRAIN_DIR}评估中运行的进程任务时间片切换信息.txt"&
#gnome-terminal --tab -t "record CPU" -- bash -c "mpstat -P ALL 1  2>&1 | tee -a ${TRAIN_DIR}评估中CPU信息.txt"&
#gnome-terminal --tab -t "record GPU" -- bash -c "nvidia-smi -lms 500  2>&1 | tee -a ${TRAIN_DIR}评估中GPU信息.txt"&
#
#sleep 2s
gnome-terminal --tab --active -t "record run.py" -- bash -ic "
${ACTIVATE_ENV}
${PY_CMD} 2>&1 | tee -a ${TRAIN_DIR}评估打印信息.txt
exec bash"&