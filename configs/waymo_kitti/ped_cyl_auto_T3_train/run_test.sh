#!/bin/bash

CONFIGS_NAME="ped_cyl_auto_T3_train"  # Need to set
ACTIVATE_ENV="conda activate nvtf1.15"  # Need to set

TRAIN_DIR="./checkpoints/waymo_kitti/${CONFIGS_NAME}/"

# --dataset_split_file to set
PY_CMD="python run_for_waymo_kitti.py \
${TRAIN_DIR} \
--test \
--dataset_root_dir ../dataset/waymo_kitti \
--dataset_split_file ../dataset/waymo_kitti/ImageSets/test.txt \
--output_dir ${TRAIN_DIR} \
--level 1"

mkdir ${TRAIN_DIR}

gnome-terminal --tab -t "record pidstat CPU" -- bash -c "pidstat 1 -l  2>&1 | tee -a ${TRAIN_DIR}测试中运行的进程CPU信息.txt"&
gnome-terminal --tab -t "record pidstat 内存" -- bash -c "pidstat 1 -r  2>&1 | tee -a ${TRAIN_DIR}测试中运行的进程内存信息.txt"&
gnome-terminal --tab -t "record pidstat 任务时间片切换" -- bash -c "pidstat 1 -w  2>&1 | tee -a ${TRAIN_DIR}测试中运行的进程任务时间片切换信息.txt"&
gnome-terminal --tab -t "record CPU" -- bash -c "mpstat -P ALL 1  2>&1 | tee -a ${TRAIN_DIR}测试中CPU信息.txt"&
gnome-terminal --tab -t "record GPU" -- bash -c "nvidia-smi -lms 500  2>&1 | tee -a ${TRAIN_DIR}测试中GPU信息.txt"&

sleep 2s
gnome-terminal --tab --active -t "record run.py" -- bash -ic "
${ACTIVATE_ENV}
${PY_CMD} 2>&1 | tee -a ${TRAIN_DIR}测试打印信息.txt
exec bash"&