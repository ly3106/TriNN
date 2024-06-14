# TriNN

If you find this code useful in your research, please consider citing our work:
```
@ARTICLE{10547414,
  author={Li, Yuanyuan and Zou, Yuan and Zhang, Xudong and Zang, Zheng and Li, Xingkun and Sun, Wenjing and Tang, Jiaqiao},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={TriNN: A Concise, Lightweight, and Fast Global Triangulation GNN for Point Cloud}, 
  year={2024},
  volume={},
  number={},
  pages={1-17},
  keywords={Point cloud compression;Kernel;Laser radar;Belts;Intelligent vehicles;Convolutional codes;Convolutional neural networks;Delaunay triangulation;Deep learning;Range belt;Range plane;Graph Neural Networks (GNNs);Object detection;Point cloud},
  doi={10.1109/TIV.2024.3409365}}
```

This repository forked and modified from [WeijingShi/Point-GNN](https://github.com/WeijingShi/Point-GNN).

This repository contains a reference implementation of original [Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Point-GNN_Graph_Neural_Network_for_3D_Object_Detection_in_a_CVPR_2020_paper.pdf), CVPR 2020. 

If you find this code useful in your research, please consider citing this work:
```
@InProceedings{Point-GNN,
author = {Shi, Weijing and Rajkumar, Ragunathan (Raj)},
title = {Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Getting Started

### Prerequisites

We use Tensorflow 1.15 for this implementation. Please [install CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) if you want GPU support.   
```
pip3 install --user tensorflow-gpu==1.15.0
```

To install other dependencies: 
```
pip3 install --user opencv-python
pip3 install --user open3d-python==0.7.0.0
pip3 install --user scikit-learn
pip3 install --user tqdm
pip3 install --user shapely
```

### KITTI Dataset

We use the KITTI 3D Object Detection dataset. Please download the dataset from the KITTI website and also download the 3DOP train/val split [here](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz). We provide extra split files for seperated classes in [splits/](splits). We recommand the following file structure:

    DATASET_ROOT_DIR
    ├── image                    #  Left color images
    │   ├── training
    |   |   └── image_2            
    │   └── testing
    |       └── image_2 
    ├── velodyne                 # Velodyne point cloud files
    │   ├── training
    |   |   └── velodyne            
    │   └── testing
    |       └── velodyne 
    ├── calib                    # Calibration files
    │   ├── training
    |   |   └──calib            
    │   └── testing
    |       └── calib 
    ├── labels                   # Training labels
    │   └── training
    |       └── label_2
    └── 3DOP_splits              # split files.
        ├── train.txt
        ├── train_car.txt
        └── ...

### Waymo Open Dataset
The Waymo Open Dataset should be converted to kitti format using [MMdetection3D v1.1.0](https://mmdetection3d.readthedocs.io/en/v1.1.0/user_guides/dataset_prepare.html), If you want to use it.

### Dataset directory
Organize your datasets into the structure outlined below. If your dataset is located in a different directory, consider creating a symbolic link to the specified directories.

    PARENT_DIR
    ├── dataset
    │   ├── kitti
    │   └── waymo_kitti
    └── TriNN
        └── ...

### Download TriNN

Clone the repository recursively:
```
git clone https://github.com/ly3106/TriNN.git --recursive
```

## Inference
### Run a checkpoint
Test on the validation split:
```
python3 run_for_kitti.py checkpoints/car_auto_T3_train/ --dataset_root_dir DATASET_ROOT_DIR --output_dir DIR_TO_SAVE_RESULTS
```
Test on the test dataset:
```
python3 run_for_kitti.py checkpoints/car_auto_T3_trainval/ --test --dataset_root_dir DATASET_ROOT_DIR --output_dir DIR_TO_SAVE_RESULTS
```

```
usage: run_for_kitti.py [-h] [-l LEVEL] [--test] [--no-box-merge] [--no-box-score]
              [--dataset_root_dir DATASET_ROOT_DIR]
              [--dataset_split_file DATASET_SPLIT_FILE]
              [--output_dir OUTPUT_DIR]
              checkpoint_path

Point-GNN inference on KITTI

positional arguments:
  checkpoint_path       Path to checkpoint

optional arguments:
  -h, --help            show this help message and exit
  -l LEVEL, --level LEVEL
                        Visualization level, 0 to disable,1 to nonblocking
                        visualization, 2 to block.Default=0
  --test                Enable test model
  --no-box-merge        Disable box merge.
  --no-box-score        Disable box score.
  --dataset_root_dir DATASET_ROOT_DIR
                        Path to KITTI dataset. Default="../dataset/kitti/"
  --dataset_split_file DATASET_SPLIT_FILE
                        Path to KITTI dataset split
                        file.Default="DATASET_ROOT_DIR/3DOP_splits/val.txt"
  --output_dir OUTPUT_DIR
                        Path to save the detection
                        resultsDefault="CHECKPOINT_PATH/eval/"
```
If you want to perform inference on the Waymo Open Dataset, you should use `run_for_waymo_kitti.py` instead of `run_for_kitti.py` in above instructions.
Alternatively, you can run the `.sh` files in each config folder to automatically execute the script and log the resource usage of hardware. The `sh` files can be found in the `configs/` directory. Before running the `sh` files, you should adjust their contents to match your environment.

### Performance
Install kitti_native_evaluation offline evaluation:
```
cd kitti_native_evaluation
cmake ./
make
```
Evaluate output results on the validation split:
```
evaluate_object_offline DATASET_ROOT_DIR/labels/training/label_2/ DIR_TO_SAVE_RESULTS
```

## Training
We put training parameters in a train_config file. To start training, we need both the train_config and config.
```
usage: train.py [-h] [--dataset_root_dir DATASET_ROOT_DIR]
                [--dataset_split_file DATASET_SPLIT_FILE]
                train_config_path config_path

Training of PointGNN

positional arguments:
  train_config_path     Path to train_config
  config_path           Path to config

optional arguments:
  -h, --help            show this help message and exit
  --dataset_root_dir DATASET_ROOT_DIR
                        Path to KITTI dataset. Default="../dataset/kitti/"
  --dataset_split_file DATASET_SPLIT_FILE
                        Path to KITTI dataset split file.Default="DATASET_ROOT
                        _DIR/3DOP_splits/train_config["train_dataset"]"
```
For example:
```
python3 train_for_kitti.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config
```
If you want to perform inference on the Waymo Open Dataset, you should use `train_for_waymo_kitti.py` instead of `train_for_kitti.py` in above instructions.
Alternatively, you can run the `.sh` files in each config folder to automatically execute the script and log the resource usage of hardware. The `sh` files can be found in the `configs/` directory. Before running the `sh` files, you should adjust their contents to match your environment.

We strongly recommand readers to view the train_config before starting the training. 
Some common parameters which you might want to change first:
```
train_dir     The directory where checkpoints and logs are stored.
train_dataset The dataset split file for training. 
NUM_GPU       The number of GPUs to use. We used two GPUs for the reference model. 
              If you want to use a single GPU, you might also need to reduce the batch size by half to save GPU memory.
              Similarly, you might want to increase the batch size if you want to utilize more GPUs. 
              Check the train.py for details.               
```
We also provide an evaluation script to evaluate the checkpoints periodically. For example:
```
python3 eval.py configs/car_auto_T3_train_eval_config 
```
You can use tensorboard to view the training and evaluation status. 
```
tensorboard --logdir=./train_dir
```
## Submission
### kitti
Reference [here](tools/README.md).
### waymo_kitti
1. Converter these KITTI-format `.txt` files to waymo `.bin` files using [waymo_kitti_converter](https://github.com/ly3106/waymo_kitti_converter)'s [tool](https://github.com/ly3106/waymo_kitti_converter?tab=readme-ov-file#convert-kitti-format-results-to-wod-format-results).
2. Converter the scores into the range [0,1] with `tanh`
```bash
conda activate openmmlab140
python tools/waymo_kitti/read_and_tanh_scores.py waymo_submission/comb-resault.bin
```
3. Evaluate locally or submit to WOD official website following [WOD library](https://github.com/waymo-research/waymo-open-dataset/tree/v1.5.0)'s [instruction](https://github.com/waymo-research/waymo-open-dataset/blob/v1.5.0/docs/quick_start.md) or [tutorial](https://github.com/waymo-research/waymo-open-dataset/blob/v1.5.0/tutorial/tutorial.ipynb).
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


