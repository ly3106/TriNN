"""This file defines the training process of Point-GNN object detection."""
"""修改日志：
train_v4.4_for_waymo_kitti直接修改DataProvider."""


import os
import sys
import time
import warnings
from datetime import datetime
import json
import argparse
import copy
from sys import getsizeof
from multiprocessing import Pool, Queue, Process

import numpy as np
import tensorflow as tf

from dataset.waymo_kitti_dataset import KittiDataset as WaymoKittiDataset
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn,\
    get_encoding_len
from models.crop_aug import CropAugSampler
from models import preprocess
from util.tf_util import average_gradients
from util.config_util import save_config, save_train_config, \
    load_train_config, load_config
from util.summary_util import write_summary_scale

from my_fun.graph_generator import generate_graph_edges_of_2D_delaunay

parser = argparse.ArgumentParser(description='Training of PointGNN')
parser.add_argument('train_config_path', type=str,
                   help='Path to train_config')
parser.add_argument('config_path', type=str,
                   help='Path to config')
parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                   help='Path to KITTI dataset. Default="../dataset/kitti/"')
parser.add_argument('--dataset_split_file', type=str,
                    default='',
                   help='Path to KITTI dataset split file.'
                   'Default="DATASET_ROOT_DIR/3DOP_splits'
                   '/train_config["train_dataset"]"')

args = parser.parse_args()
train_config = load_train_config(args.train_config_path)
DATASET_DIR = args.dataset_root_dir
if args.dataset_split_file == '':
    DATASET_SPLIT_FILE = os.path.join(DATASET_DIR,
        './3DOP_splits/'+train_config['train_dataset'])
else:
    DATASET_SPLIT_FILE = args.dataset_split_file
config_complete = load_config(args.config_path)
if 'train' in config_complete:
    config = config_complete['train']
else:
    config = config_complete
# input function ==============================================================
# 配置数据路径
dataset = WaymoKittiDataset(
    os.path.join(DATASET_DIR, 'training/image_0/'), # image/training/image_2
    os.path.join(DATASET_DIR, 'training/velodyne/'), # velodyne/training/velodyne/
    os.path.join(DATASET_DIR, 'training/calib/'), # calib/training/calib/
    os.path.join(DATASET_DIR, 'training/label_all'), # labels/training/label_2
    DATASET_SPLIT_FILE,
    num_classes=config['num_classes'])
NUM_CLASSES = dataset.num_classes

# 配置测试样本数量
if 'NUM_TEST_SAMPLE' not in train_config:
    NUM_TEST_SAMPLE = dataset.num_files
else:
    if train_config['NUM_TEST_SAMPLE'] < 0:
        NUM_TEST_SAMPLE = dataset.num_files
    else:
        NUM_TEST_SAMPLE = train_config['NUM_TEST_SAMPLE']

# 配置box编码和解码方法
BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn = get_box_decoding_fn(config['box_encoding_method'])

# 配置数据增强方式
aug_fn = preprocess.get_data_aug(train_config['data_aug_configs'])

# 但由于所有配置文件均未配置相关参数，故实际该函数未被使用
if 'crop_aug' in train_config:
    sampler = CropAugSampler(train_config['crop_aug']['crop_filename'])

def fetch_data(frame_idx):
    # cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx,
    #     config['downsample_by_voxel_size'])
    cam_rgb_points = cam_points = dataset.get_cam_points(frame_idx, config['downsample_by_voxel_size'])
    box_label_list = dataset.get_label(frame_idx)
    if 'crop_aug' in train_config:
        cam_rgb_points, box_label_list = sampler.crop_aug(cam_rgb_points,
            box_label_list,
            sample_rate=train_config['crop_aug']['sample_rate'],
            parser_kwargs=train_config['crop_aug']['parser_kwargs'])
    cam_rgb_points, box_label_list = aug_fn(cam_rgb_points, box_label_list)
    graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])
    (vertex_coord_list, keypoint_indices_list, edges_list) = \
        graph_generate_fn(cam_rgb_points.xyz, **config['graph_gen_kwargs'])

    if config['input_features'] == 'irgb':
        input_v = cam_rgb_points.attr
    elif config['input_features'] == '0rgb':
        input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)),
            cam_rgb_points.attr[:, 1:]])
    elif config['input_features'] == '0000':
        input_v = np.zeros_like(cam_rgb_points.attr)
    elif config['input_features'] == 'i000':
        input_v = np.hstack([cam_rgb_points.attr[:, [0]],
            np.zeros((cam_rgb_points.attr.shape[0], 3))])
    elif config['input_features'] == 'i':
        input_v = cam_rgb_points.attr[:, [0]]
    elif config['input_features'] == '0':
        input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))
    last_layer_graph_level = config['model_kwargs'][
        'layer_configs'][-1]['graph_level']
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1] # 最后一层直接用于预测结果的点云
    if config['label_method'] == 'yaw':
        cls_labels, boxes_3d, valid_boxes, label_map = \
            dataset.assign_classaware_label_to_points(box_label_list,
            last_layer_points_xyz,
            expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    if config['label_method'] == 'Car':
        cls_labels, boxes_3d, valid_boxes, label_map = \
            dataset.assign_classaware_car_label_to_points(box_label_list,
            last_layer_points_xyz,
            expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    if config['label_method'] == 'Pedestrian_and_Cyclist':
        (cls_labels, boxes_3d, valid_boxes, label_map) =\
            dataset.assign_classaware_ped_and_cyc_label_to_points(
            box_label_list, last_layer_points_xyz,
            expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    encoded_boxes = box_encoding_fn(cls_labels, last_layer_points_xyz,
        boxes_3d, label_map)
    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]
    cls_labels = cls_labels.astype(np.int32)
    encoded_boxes = encoded_boxes.astype(np.float32)
    valid_boxes = valid_boxes.astype(np.float32)
    return(input_v, vertex_coord_list, keypoint_indices_list, edges_list,
        cls_labels, encoded_boxes, valid_boxes)

def batch_data(batch_list):
    N_input_v, N_vertex_coord_list, N_keypoint_indices_list, N_edges_list,\
    N_cls_labels, N_encoded_boxes, N_valid_boxes = zip(*batch_list)
    batch_size = len(batch_list)
    level_num = len(N_vertex_coord_list[0])
    batched_keypoint_indices_list = []
    batched_edges_list = []
    for level_idx in range(level_num-1):
        centers = []
        vertices = []
        point_counter = 0
        center_counter = 0
        for batch_idx in range(batch_size):
            centers.append(
                N_keypoint_indices_list[batch_idx][level_idx]+point_counter)
            vertices.append(np.hstack(
                [N_edges_list[batch_idx][level_idx][:,[0]]+point_counter,
                 N_edges_list[batch_idx][level_idx][:,[1]]+center_counter]))
            point_counter += N_vertex_coord_list[batch_idx][level_idx].shape[0]
            center_counter += \
                N_keypoint_indices_list[batch_idx][level_idx].shape[0]
        batched_keypoint_indices_list.append(np.vstack(centers))
        batched_edges_list.append(np.vstack(vertices))
    batched_vertex_coord_list = []
    for level_idx in range(level_num):
        points = []
        counter = 0
        for batch_idx in range(batch_size):
            points.append(N_vertex_coord_list[batch_idx][level_idx])
        batched_vertex_coord_list.append(np.vstack(points))
    batched_input_v = np.vstack(N_input_v)
    batched_cls_labels = np.vstack(N_cls_labels)
    batched_encoded_boxes = np.vstack(N_encoded_boxes)
    batched_valid_boxes = np.vstack(N_valid_boxes)
    return (batched_input_v, batched_vertex_coord_list,
        batched_keypoint_indices_list, batched_edges_list, batched_cls_labels,
        batched_encoded_boxes, batched_valid_boxes)

# model =======================================================================
if 'COPY_PER_GPU' in train_config:
    COPY_PER_GPU = train_config['COPY_PER_GPU'] # 每个GPU的同时处理的数量？该参数在配置文件中没有写，故直接执行else
else:
    COPY_PER_GPU = 1
NUM_GPU = train_config['NUM_GPU']
input_tensor_sets = []
for gi in range(NUM_GPU): #对每个GPU进行任务配置 # set assignments for each GPU
    with tf.device('/gpu:%d'%gi): # 对当前循环GPU进行任务配置 # set assignments for the GPU in current loop 
        for cp_idx in range(COPY_PER_GPU): # 对当前循环GPU的每个批处理进行配置 # set each batch of the GPU in current loop
            # 初始化集成特征点的类型，原始配置文件采用的是i，但作者论文中采用的应该是降采样产生的特征点
            # 猜测i指反射率，rgb为三原色，0为填充对应维度0
            if config['input_features'] == 'irgb':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 4])
            elif config['input_features'] == 'rgb':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 3])
            elif config['input_features'] == '0000':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 4])
            elif config['input_features'] == 'i000':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 4])
            elif config['input_features'] == 'i':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 1])
            elif config['input_features'] == '0':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 1])

            # 初始化图节点容器？节点类型为xyz？
            t_vertex_coord_list = [
                tf.placeholder(dtype=tf.float32, shape=[None, 3])]
            for _ in range(len(config['graph_gen_kwargs']['level_configs'])):# 配置文件中配置几层，这里就需要几个节点张量
                t_vertex_coord_list.append(
                    tf.placeholder(dtype=tf.float32, shape=[None, 3]))

            # 初始化图边容器
            t_edges_list = []
            for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
                t_edges_list.append(
                    tf.placeholder(dtype=tf.int32, shape=[None, None]))

            # 初始化关键点下标容器？
            t_keypoint_indices_list = []
            for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
                t_keypoint_indices_list.append(
                    tf.placeholder(dtype=tf.int32, shape=[None, 1]))

            t_class_labels = tf.placeholder(dtype=tf.int32, shape=[None, 1]) # 初始化分类标签
            t_encoded_gt_boxes = tf.placeholder(
                dtype=tf.float32, shape=[None, 1, BOX_ENCODING_LEN]) # 初始化ground truth框
            t_valid_gt_boxes = tf.placeholder(
                dtype=tf.float32, shape=[None, 1, 1]) # model.loss()函数中有解释其意义
            t_is_training = tf.placeholder(dtype=tf.bool, shape=[])

            # 初始化模型
            model = get_model(config['model_name'])(num_classes=NUM_CLASSES,
                box_encoding_len=BOX_ENCODING_LEN, mode='train',
                **config['model_kwargs'])
            t_logits, t_pred_box = model.predict(
                t_initial_vertex_features, t_vertex_coord_list,
                t_keypoint_indices_list, t_edges_list, t_is_training)
            t_probs = model.postprocess(t_logits)
            t_predictions = tf.argmax(t_probs, axis=-1, output_type=tf.int32)
            t_loss_dict = model.loss(t_logits, t_class_labels, t_pred_box,
                t_encoded_gt_boxes, t_valid_gt_boxes, **config['loss'])
            t_cls_loss = t_loss_dict['cls_loss']
            t_loc_loss = t_loss_dict['loc_loss']
            t_reg_loss = t_loss_dict['reg_loss']
            t_num_endpoint = t_loss_dict['num_endpoint']
            t_num_valid_endpoint = t_loss_dict['num_valid_endpoint']
            t_classwise_loc_loss = t_loss_dict['classwise_loc_loss']
            t_total_loss = t_cls_loss + t_loc_loss + t_reg_loss
            input_tensor_sets.append(
                {'t_initial_vertex_features': t_initial_vertex_features,
                 't_vertex_coord_list': t_vertex_coord_list,
                 't_edges_list':t_edges_list,
                 't_keypoint_indices_list': t_keypoint_indices_list,
                 't_class_labels': t_class_labels,
                 't_encoded_gt_boxes': t_encoded_gt_boxes,
                 't_valid_gt_boxes': t_valid_gt_boxes,
                 't_is_training': t_is_training,
                 't_logits': t_logits,
                 't_pred_box': t_pred_box,
                 't_probs': t_probs,
                 't_predictions': t_predictions,
                 't_cls_loss': t_cls_loss,
                 't_loc_loss': t_loc_loss,
                 't_reg_loss': t_reg_loss,
                 't_num_endpoint': t_num_endpoint,
                 't_num_valid_endpoint': t_num_valid_endpoint,
                 't_classwise_loc_loss': t_classwise_loc_loss,
                 't_total_loss': t_total_loss
                 })

if 'unify_copies' in train_config:
    if train_config['unify_copies']:
        # re-weight loss for the number of end points
        print('Set to unify copies in different GPU as if its a single copy')
        total_num_endpoints = tf.reduce_sum([t['t_num_endpoint']
            for t in input_tensor_sets])
        total_num_valid_endpoints = tf.reduce_sum([t['t_num_valid_endpoint']
            for t in input_tensor_sets])
        for ti in range(len(input_tensor_sets)):
            weight = tf.div_no_nan(
                tf.cast(len(input_tensor_sets)*input_tensor_sets[ti][
                    't_num_endpoint'], tf.float32),
                tf.cast(total_num_endpoints, tf.float32))
            weight = tf.cast(weight, tf.float32)
            valid_weight = tf.div_no_nan(
                tf.cast(len(input_tensor_sets)*input_tensor_sets[ti][
                    't_num_valid_endpoint'], tf.float32),
                tf.cast(total_num_valid_endpoints, tf.float32))
            valid_weight = tf.cast(valid_weight, tf.float32)
            input_tensor_sets[ti]['t_cls_loss'] *= weight
            input_tensor_sets[ti]['t_loc_loss'] *= valid_weight
            input_tensor_sets[ti]['t_total_loss'] = \
                input_tensor_sets[ti]['t_cls_loss']\
                +input_tensor_sets[ti]['t_loc_loss']\
                +input_tensor_sets[ti]['t_reg_loss']

t_cls_loss_cross_gpu = tf.reduce_mean([t['t_cls_loss']
    for t in input_tensor_sets])
t_loc_loss_cross_gpu = tf.reduce_mean([t['t_loc_loss']
    for t in input_tensor_sets])
t_reg_loss_cross_gpu = tf.reduce_mean([t['t_reg_loss']
    for t in input_tensor_sets])
t_total_loss_cross_gpu = tf.reduce_mean([t['t_total_loss']
    for t in input_tensor_sets])

t_class_labels = input_tensor_sets[0]['t_class_labels']
t_predictions = input_tensor_sets[0]['t_predictions']
t_probs = input_tensor_sets[0]['t_probs']

t_classwise_loc_loss_update_ops = {}
for class_idx in range(NUM_CLASSES):
    for bi in range(BOX_ENCODING_LEN):
        classwise_loc_loss_ind =tf.reduce_sum(
            [input_tensor_sets[gi]['t_classwise_loc_loss'][class_idx][bi]
                for gi in range(len(input_tensor_sets))])
        t_mean_loss, t_mean_loss_op = tf.metrics.mean(
            classwise_loc_loss_ind,
            name=('loc_loss_cls_%d_box_%d'%(class_idx, bi)))
        t_classwise_loc_loss_update_ops[
            ('loc_loss_cls_%d_box_%d'%(class_idx, bi))] = t_mean_loss_op
    classwise_loc_loss =tf.reduce_sum(
        [input_tensor_sets[gi]['t_classwise_loc_loss'][class_idx]
            for gi in range(len(input_tensor_sets))])
    t_mean_loss, t_mean_loss_op = tf.metrics.mean(
        classwise_loc_loss,
        name=('loc_loss_cls_%d'%class_idx))
    t_classwise_loc_loss_update_ops[
        ('loc_loss_cls_%d'%class_idx)] = t_mean_loss_op

# metrics测试训练结果
t_recall_update_ops = {}
for class_idx in range(NUM_CLASSES):
    t_recall, t_recall_update_op = tf.metrics.recall(
        tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
        tf.equal(t_predictions, tf.constant(class_idx, tf.int32)),
        name=('recall_%d'%class_idx))
    t_recall_update_ops[('recall_%d'%class_idx)] = t_recall_update_op

t_precision_update_ops = {}
for class_idx in range(NUM_CLASSES):
    t_precision, t_precision_update_op = tf.metrics.precision(
        tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
        tf.equal(t_predictions, tf.constant(class_idx, tf.int32)),
        name=('precision_%d'%class_idx))
    t_precision_update_ops[('precision_%d'%class_idx)] = t_precision_update_op

t_mAP_update_ops = {}
for class_idx in range(NUM_CLASSES):
    t_mAP, t_mAP_update_op = tf.metrics.auc(
        tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
        t_probs[:, class_idx],
        num_thresholds=200,
        curve='PR',
        name=('mAP_%d'%class_idx),
        summation_method='careful_interpolation')
    t_mAP_update_ops[('mAP_%d'%class_idx)] = t_mAP_update_op

t_mean_cls_loss, t_mean_cls_loss_op = tf.metrics.mean(
    t_cls_loss_cross_gpu,
    name='mean_cls_loss')
t_mean_loc_loss, t_mean_loc_loss_op = tf.metrics.mean(
    t_loc_loss_cross_gpu,
    name='mean_loc_loss')
t_mean_reg_loss, t_mean_reg_loss_op = tf.metrics.mean(
    t_reg_loss_cross_gpu,
    name='mean_reg_loss')
t_mean_total_loss, t_mean_total_loss_op = tf.metrics.mean(
    t_total_loss_cross_gpu,
    name='mean_total_loss')

metrics_update_ops = {
    'cls_loss': t_mean_cls_loss_op,
    'loc_loss': t_mean_loc_loss_op,
    'reg_loss': t_mean_reg_loss_op,
    'total_loss': t_mean_total_loss_op,}
metrics_update_ops.update(t_recall_update_ops)
metrics_update_ops.update(t_precision_update_ops)
metrics_update_ops.update(t_mAP_update_ops)
metrics_update_ops.update(t_classwise_loc_loss_update_ops)

# optimizers ================================================================
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
t_learning_rate = tf.train.exponential_decay(train_config['initial_lr'],
    global_step, train_config['decay_step'], train_config['decay_factor'],
    staircase=train_config.get('is_staircase', True))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer_dict = {
    'sgd': tf.train.GradientDescentOptimizer,
    'momentum': tf.train.MomentumOptimizer,
    'rmsprop':  tf.train.RMSPropOptimizer,
    'adam': tf.train.AdamOptimizer,
}
optimizer_kwargs_dict = {
    'sgd': {},
    'momentum': {'momentum': 0.9},
    'rmsprop':  {'momentum': 0.9, 'decay': 0.9, 'epsilon': 1.0},
    'adam': {}
}
optimizer_class = optimizer_dict[train_config['optimizer']]
optimizer_kwargs = optimizer_kwargs_dict[train_config['optimizer']]
if 'optimizer_kwargs' in train_config:
    optimizer_kwargs.update(train_config['optimizer_kwargs'])
optimizer = optimizer_class(t_learning_rate, **optimizer_kwargs)
grads_cross_gpu = []
with tf.control_dependencies(update_ops):
    for gi in range(NUM_GPU):
        with tf.device('/gpu:%d'%gi):
            grads = optimizer.compute_gradients(
                input_tensor_sets[gi]['t_total_loss'])
            grads_cross_gpu.append(grads)
grads_cross_gpu = average_gradients(grads_cross_gpu)
train_op = optimizer.apply_gradients(grads_cross_gpu, global_step=global_step)
fetches = {
    'train_op': train_op,
    'step': global_step,
    'learning_rate': t_learning_rate,
}
fetches.update(metrics_update_ops)

class DataProvider(object):
    """This class provides input data to training.
    It has option to load dataset in memory so that preprocessing does not
    repeat every time.
    Note, if there is randomness inside graph creation, dataset should be
    reloaded.
    """
    def __init__(self, fetch_data, batch_data, load_dataset_to_mem=True,
        load_dataset_every_N_time=1, capacity=1, num_workers=1, preload_list=[],
        async_load_rate=1.0, result_pool_limit=64): # 10000
        self._fetch_data = fetch_data
        self._batch_data = batch_data
        self._buffer = {}
        self._results = {}
        self._load_dataset_to_mem = load_dataset_to_mem
        self._load_every_N_time = load_dataset_every_N_time
        self._capacity = capacity
        self._worker_pool = Pool(processes=num_workers)
        self._preload_list = preload_list
        self._preload_list_to_index_dict = {}
        self._async_load_rate = async_load_rate
        self._result_pool_limit = result_pool_limit
        if len(self._preload_list) > 0:
            self.preload(self._preload_list)

    def preload(self, frame_idx_list):
        """async load dataset into memory."""
        for frame_idx in frame_idx_list:
            result = self._worker_pool.apply_async(
                self._fetch_data, (frame_idx,))
            self._results[frame_idx] = result

    def set_load_list(self, load_list):
        """Sets the list of data to be preloaded."""
        self._preload_list = load_list
        self._preload_list_to_index_dict = {value: index for index, value in enumerate(self._preload_list)}
        self._preload_generator = self._manage_preload_generator()
        next(self._preload_generator)  # Start the preload process

    def _manage_preload_generator(self):
        """A generator that manages preloading data as needed."""
        for frame_idx in self._preload_list:
            if len(self._results) >= self._result_pool_limit:
                yield  # Pause preloading if the result pool is full
            if frame_idx not in self._results:
                result = self._worker_pool.apply_async(self._fetch_data, (frame_idx,))
                self._results[frame_idx] = result
        yield  # Final yield to allow generator to complete

    def check_preload(self):
        """Resume preloading if possible."""
        if self._preload_generator:
            try:
                next(self._preload_generator)
            except StopIteration:
                self._preload_generator = None  # Preloading complete

    def del_past_frames(self, frame_idx):
        current_frame_idx_list_idx = self._preload_list_to_index_dict[frame_idx]
        start_idx = current_frame_idx_list_idx + 1
        if start_idx >= len(self._preload_list): return
        end_idx = min([start_idx+self._result_pool_limit, len(self._preload_list)])
        future_frame_idx_list = self._preload_list[start_idx:end_idx]
        # 删除不在future_frame_idx_list中的keys
        keys_to_delete = [key for key in self._results if key not in future_frame_idx_list]
        for key in keys_to_delete:
            del self._results[key]


    def async_load(self, frame_idx):
        """async load a data into memory"""
        if frame_idx in self._results:
            data = self._results[frame_idx].get()
            del self._results[frame_idx]
            self.check_preload()  # Check if we can resume preloading
        else:
            data = self._fetch_data(frame_idx)
            warnings.warn("DataProvider did not prelode frame_idx=%d" % frame_idx)
            sys.stdout.flush()  # 强制刷新缓冲区
            del_past_frames(frame_idx)
        if np.random.random() < self._async_load_rate:
            if len(self._results) < self._result_pool_limit:
                result = self._worker_pool.apply_async(
                    self._fetch_data, (frame_idx,))
                self._results[frame_idx] = result
        return data

    def provide(self, frame_idx):
        if self._load_dataset_to_mem:
            if self._load_every_N_time >= 1:
                extend_frame_idx = frame_idx+np.random.choice(
                    self._capacity)*NUM_TEST_SAMPLE
                if extend_frame_idx not in self._buffer:
                    data = self.async_load(frame_idx)
                    self._buffer[extend_frame_idx] = (data, 0)
                data, ctr = self._buffer[extend_frame_idx]
                if ctr == self._load_every_N_time:
                    data = self.async_load(frame_idx)
                    self._buffer[extend_frame_idx] = (data, 0)
                data, ctr = self._buffer[extend_frame_idx]
                self._buffer[extend_frame_idx] = (data, ctr+1)
                return data
            else:
                # do not buffer
                return self.async_load(frame_idx)
        else:
            return self._fetch_data(frame_idx)

    def provide_batch(self, frame_idx_list):
        batch_list = []
        for frame_idx in frame_idx_list:
            batch_list.append(self.provide(frame_idx))
        return self._batch_data(batch_list)

data_provider = DataProvider(fetch_data, batch_data,
    load_dataset_to_mem=train_config['load_dataset_to_mem'],
    load_dataset_every_N_time=train_config['load_dataset_every_N_time'],
    capacity=train_config['capacity'],
    num_workers=train_config['num_load_dataset_workers'],
    preload_list=[], # list(range(NUM_TEST_SAMPLE))
    async_load_rate=0.)


# Training session ==========================================================
batch_size = train_config.get('batch_size', 1)
print('batch size=' + str(batch_size))
saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
graph = tf.get_default_graph()
if train_config['gpu_memusage'] < 0:
    gpu_options = tf.GPUOptions(allow_growth=True)
else:
    if train_config['gpu_memusage'] < -10:
        gpu_options = tf.GPUOptions()
    else:
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=train_config['gpu_memusage'])

# 从检查点恢复或初始化训练状态，包括frame_idx_list
def restore_training_batch_state():
    start_batch_idx = 0
    frame_idx_list = None
    # 尝试从JSON文件恢复额外的状态信息，包括frame_idx_list
    state_file = os.path.join(train_config['train_dir'], 'training_state.json')
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            training_state = json.load(f)
            start_batch_idx = training_state['batch_idx']
            frame_idx_list = training_state.get('frame_idx_list', None)
            if frame_idx_list is not None:
                frame_idx_list = np.array(frame_idx_list)  # 将列表转换回NumPy数组
    return start_batch_idx, frame_idx_list

# 保存当前训练状态
def save_training_batch_state(batch_idx, frame_idx_list):
    # 保存状态信息为JSON格式
    training_state = {
        'batch_idx': batch_idx,
        'frame_idx_list': frame_idx_list.tolist()  # 将NumPy数组转换为列表
    }
    with open(os.path.join(train_config['train_dir'], 'training_state.json'), 'w') as f:
        json.dump(training_state, f)
def all_elements_are_valid(arr):
    # 检查数组中的元素是否都不是NaN且不是无穷大
    # 如果arr是列表，检查列表中的每个数组
    if isinstance(arr, list):
        return all(np.all(np.logical_not(np.logical_or(np.isnan(a), np.isinf(a)))) for a in arr)
    # 否则，假设arr是单个NumPy数组
    else:
        return np.all(np.logical_not(np.logical_or(np.isnan(arr), np.isinf(arr))))

batch_ctr = 0
batch_gradient_list = []
with tf.Session(graph=graph,
    config=tf.ConfigProto(
    allow_soft_placement=True, gpu_options=gpu_options,)) as sess:
    sess.run(tf.variables_initializer(tf.global_variables()))
    states = tf.train.get_checkpoint_state(train_config['train_dir'])
    if states is not None:
        print('Restore from checkpoint %s' % states.model_checkpoint_path)
        saver.restore(sess, states.model_checkpoint_path)
        saver.recover_last_checkpoints(states.all_model_checkpoint_paths)
        start_batch_idx, restored_frame_idx_list = restore_training_batch_state()
    else:
        start_batch_idx = 0
        restored_frame_idx_list = None
    previous_step = sess.run(global_step)
    local_variables_initializer = tf.variables_initializer(tf.local_variables())
    for epoch_idx in range((previous_step*batch_size)//NUM_TEST_SAMPLE,
    train_config['max_epoch']):
        sess.run(local_variables_initializer)
        start_time = time.time()
        if restored_frame_idx_list is None:
            assert start_batch_idx == 0 or start_batch_idx is None, "Start_batch_idx wasn't reset!"
            frame_idx_list = np.random.permutation(NUM_TEST_SAMPLE) #将训练样本序号随机打乱 # shuffle sequence of training sample
            start_batch_idx = 0
        else:
            assert start_batch_idx is not None, "Start_batch_idx wasn't set!"
            frame_idx_list = restored_frame_idx_list
            start_batch_idx = start_batch_idx + batch_size // (COPY_PER_GPU * NUM_GPU)
            if start_batch_idx >= NUM_TEST_SAMPLE - batch_size + 1:
                continue
            print('Continue training from batch_idx %d of epoch %d' % (start_batch_idx, epoch_idx))

        # for batch_idx in range(start_batch_idx, NUM_TEST_SAMPLE - batch_size + 1, batch_size):  # 0
        batch_idx = start_batch_idx
        data_provider.set_load_list(frame_idx_list[batch_idx:])
        while batch_idx < NUM_TEST_SAMPLE - batch_size + 1:
            mid_time = time.time()
            device_batch_size = batch_size // (COPY_PER_GPU * NUM_GPU)
            total_feed_dict = {}
            valid_boxes_found = False
            for gi in range(COPY_PER_GPU * NUM_GPU):  # 对每个GPU的拷贝进行处理
                valid_boxes_found = False
                while batch_idx + device_batch_size <= NUM_TEST_SAMPLE:
                    batch_frame_idx_list = frame_idx_list[
                                           batch_idx:\
                                           batch_idx + device_batch_size]
                    input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
                        cls_labels, encoded_boxes, valid_boxes \
                        = data_provider.provide_batch(batch_frame_idx_list)
                    # (中)检查输入数组是否有意义 # Check whether input arrarys is valid
                    if not all_elements_are_valid(input_v): warnings.warn('input_v include invalid value')
                    if not all_elements_are_valid(vertex_coord_list): warnings.warn('vertex_coord_list include invalid value')
                    if not all_elements_are_valid(encoded_boxes): warnings.warn('encoded_boxes include invalid value')
                    if not all_elements_are_valid(valid_boxes): warnings.warn('valid_boxes include invalid value')
                    if np.any(valid_boxes != 0.):
                        valid_boxes_found = True
                        print('There are valid boxes in frames range from %d, and their idx is [%d, %d]' \
                              % (frame_idx_list[batch_idx], batch_idx, batch_idx + device_batch_size))
                        batch_idx += device_batch_size
                        sys.stdout.flush()  # 强制刷新缓冲区
                        break
                    else:
                        warnings.warn('There are no valid boxes in frames range from %d, and their idx is [%d, %d]' \
                              % (frame_idx_list[batch_idx], batch_idx , batch_idx + device_batch_size))
                        batch_idx += device_batch_size
                    sys.stdout.flush()  # 强制刷新缓冲区
                if valid_boxes_found == False: break

                t_initial_vertex_features = \
                    input_tensor_sets[gi]['t_initial_vertex_features']
                t_class_labels = input_tensor_sets[gi]['t_class_labels']
                t_encoded_gt_boxes = input_tensor_sets[gi]['t_encoded_gt_boxes']
                t_valid_gt_boxes = input_tensor_sets[gi]['t_valid_gt_boxes']
                t_is_training = input_tensor_sets[gi]['t_is_training']
                t_edges_list = input_tensor_sets[gi]['t_edges_list']
                t_keypoint_indices_list = \
                    input_tensor_sets[gi]['t_keypoint_indices_list']
                t_vertex_coord_list = \
                    input_tensor_sets[gi]['t_vertex_coord_list']
                feed_dict = {
                    t_initial_vertex_features: input_v,
                    t_class_labels: cls_labels,
                    t_encoded_gt_boxes: encoded_boxes,
                    t_valid_gt_boxes: valid_boxes,
                    t_is_training: True,
                }
                feed_dict.update(dict(zip(t_edges_list, edges_list)))
                feed_dict.update(
                    dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
                feed_dict.update(
                    dict(zip(t_vertex_coord_list, vertex_coord_list)))
                total_feed_dict.update(feed_dict)
        # for batch_idx in range(start_batch_idx, NUM_TEST_SAMPLE-batch_size+1, batch_size): # 0
        #     mid_time = time.time()
        #     device_batch_size = batch_size//(COPY_PER_GPU*NUM_GPU)
        #     total_feed_dict = {}
        #     for gi in range(COPY_PER_GPU*NUM_GPU): #对每个GPU的拷贝进行处理
        #         batch_frame_idx_list = frame_idx_list[
        #             batch_idx+\
        #             gi*device_batch_size:batch_idx+(gi+1)*device_batch_size]
        #         input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
        #         cls_labels, encoded_boxes, valid_boxes \
        #             = data_provider.provide_batch(batch_frame_idx_list)
        #         t_initial_vertex_features = \
        #             input_tensor_sets[gi]['t_initial_vertex_features']
        #         t_class_labels = input_tensor_sets[gi]['t_class_labels']
        #         t_encoded_gt_boxes = input_tensor_sets[gi]['t_encoded_gt_boxes']
        #         t_valid_gt_boxes = input_tensor_sets[gi]['t_valid_gt_boxes']
        #         t_is_training = input_tensor_sets[gi]['t_is_training']
        #         t_edges_list = input_tensor_sets[gi]['t_edges_list']
        #         t_keypoint_indices_list = \
        #             input_tensor_sets[gi]['t_keypoint_indices_list']
        #         t_vertex_coord_list = \
        #             input_tensor_sets[gi]['t_vertex_coord_list']
        #         feed_dict = {
        #             t_initial_vertex_features: input_v,
        #             t_class_labels: cls_labels,
        #             t_encoded_gt_boxes: encoded_boxes,
        #             t_valid_gt_boxes: valid_boxes,
        #             t_is_training: True,
        #         }
        #         feed_dict.update(dict(zip(t_edges_list, edges_list)))
        #         feed_dict.update(
        #             dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
        #         feed_dict.update(
        #             dict(zip(t_vertex_coord_list, vertex_coord_list)))
        #         total_feed_dict.update(feed_dict)
            if valid_boxes_found == False: break
            if train_config.get('is_pseudo_batch', False):
                tf_gradient = [g for g, v in grads_cross_gpu]
                batch_gradient = sess.run(tf_gradient,
                    feed_dict=total_feed_dict)
                batch_gradient_list.append(batch_gradient)
                if batch_ctr % train_config['pseudo_batch_factor'] == 0:
                    batch_gradient_list = list(zip(*batch_gradient_list))
                    batch_gradient = [batch_gradient_list[ggi][0]
                        for ggi in range(len(batch_gradient_list)) ]
                    for ggi in range(len(batch_gradient_list)):
                        for pi in range(1, len(batch_gradient_list[ggi])):
                            batch_gradient[ggi] += batch_gradient_list[ggi][pi]
                    total_feed_dict.update(
                        dict(zip(tf_gradient, batch_gradient)))
                    results = sess.run(train_op, feed_dict=total_feed_dict)
                    batch_gradient_list = []
                batch_ctr += 1
            else:
                results = sess.run(fetches, feed_dict=total_feed_dict) # 进行梯度下降更新
            if 'max_steps' in train_config and train_config['max_steps'] > 0:
                if results['step'] >= train_config['max_steps']:
                    checkpoint_path = os.path.join(train_config['train_dir'],
                        train_config['checkpoint_path'])
                    config_path = os.path.join(train_config['train_dir'],
                        train_config['config_path'])
                    train_config_path = os.path.join(train_config['train_dir'],
                        'train_config')
                    print("datetime is: {}, and timestamp is: {}"\
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), time.time()))
                    print('reach the max_steps, save checkpoint at step %d to %s'
                        % (results['step'], checkpoint_path))
                    saver.save(sess, checkpoint_path,
                        latest_filename='checkpoint',
                        global_step=results['step'])
                    save_config(config_path, config_complete)
                    save_train_config(train_config_path, train_config)
                    save_training_batch_state(batch_idx, frame_idx_list)
                    raise SystemExit
            # (中)多个step保存一次 # (EN) Save one time after several steps =========
            if results['step'] % 20 == 0:
                print("datetime is: {}, and timestamp is: {}" \
                      .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), time.time()))
                print('STEP: %d, epoch_idx: %d, lr: %f, 1 batch time cost: %f'
                      % (results['step'], epoch_idx, results['learning_rate'],
                         time.time() - mid_time))
                print('cls:%f, loc:%f, reg:%f, loss: %f'
                      % (results['cls_loss'], results['loc_loss'], results['reg_loss'],
                         results['total_loss']))
                for class_idx in range(NUM_CLASSES):
                    print('Class_%d: recall=%f, prec=%f, mAP=%f, loc=%f'
                          % (class_idx,
                             results['recall_%d' % class_idx],
                             results['precision_%d' % class_idx],
                             results['mAP_%d' % class_idx],
                             results['loc_loss_cls_%d' % class_idx]))
                    print("         x=%.4f y=%.4f z=%.4f l=%.4f h=%.4f w=%.4f y=%.4f"
                          % (
                              results['loc_loss_cls_%d_box_%d' % (class_idx, 0)],
                              results['loc_loss_cls_%d_box_%d' % (class_idx, 1)],
                              results['loc_loss_cls_%d_box_%d' % (class_idx, 2)],
                              results['loc_loss_cls_%d_box_%d' % (class_idx, 3)],
                              results['loc_loss_cls_%d_box_%d' % (class_idx, 4)],
                              results['loc_loss_cls_%d_box_%d' % (class_idx, 5)],
                              results['loc_loss_cls_%d_box_%d' % (class_idx, 6)]),
                          )

                # add summaries ====================================================
                for key in metrics_update_ops:
                    write_summary_scale(key, results[key], results['step'],
                                        train_config['train_dir'])
                write_summary_scale('learning rate', results['learning_rate'],
                                    results['step'], train_config['train_dir'])
                # save checkpoint ==================================================
                checkpoint_path = os.path.join(train_config['train_dir'],
                                               train_config['checkpoint_path'])
                config_path = os.path.join(train_config['train_dir'],
                                           train_config['config_path'])
                train_config_path = os.path.join(train_config['train_dir'],
                                                 'train_config')
                print('save checkpoint at step %d to %s'
                      % (results['step'], checkpoint_path))
                sys.stdout.flush()  # 强制刷新缓冲区
                saver.save(sess, checkpoint_path,
                           latest_filename='checkpoint',
                           global_step=results['step'])
                save_config(config_path, config_complete)
                save_train_config(train_config_path, train_config)
                save_training_batch_state(batch_idx, frame_idx_list)

        print("datetime is: {}, and timestamp is: {}" .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), time.time()))
        print('STEP: %d, epoch_idx: %d, lr: %f, time cost: %f'
            % (results['step'], epoch_idx, results['learning_rate'],
            time.time()-start_time))
        print('cls:%f, loc:%f, reg:%f, loss: %f'
            % (results['cls_loss'], results['loc_loss'], results['reg_loss'],
            results['total_loss']))
        for class_idx in range(NUM_CLASSES):
            print('Class_%d: recall=%f, prec=%f, mAP=%f, loc=%f'
                % (class_idx,
                results['recall_%d'%class_idx],
                results['precision_%d'%class_idx],
                results['mAP_%d'%class_idx],
                results['loc_loss_cls_%d'%class_idx]))
            print("         x=%.4f y=%.4f z=%.4f l=%.4f h=%.4f w=%.4f y=%.4f"
            %(
            results['loc_loss_cls_%d_box_%d'%(class_idx, 0)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 1)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 2)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 3)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 4)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 5)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 6)]),
            )

        # add summaries ====================================================
        for key in metrics_update_ops:
            write_summary_scale(key, results[key], results['step'],
                train_config['train_dir'])
        write_summary_scale('learning rate', results['learning_rate'],
            results['step'], train_config['train_dir'])
        # save checkpoint ==================================================
        if (epoch_idx + 1) % train_config['save_every_epoch'] == 0:
            checkpoint_path = os.path.join(train_config['train_dir'],
                train_config['checkpoint_path'])
            config_path = os.path.join(train_config['train_dir'],
                train_config['config_path'])
            train_config_path = os.path.join(train_config['train_dir'],
                'train_config')
            print('save checkpoint at epoch %d to %s'
                % (epoch_idx, checkpoint_path))
            sys.stdout.flush()  # 强制刷新缓冲区
            saver.save(sess, checkpoint_path,
                latest_filename='checkpoint',
                global_step=results['step'])
            save_config(config_path, config_complete)
            save_train_config(train_config_path, train_config)
            save_training_batch_state(batch_idx, frame_idx_list)
            restored_frame_idx_list = None # 如果当前epoch正常完成，确保下一个epoch不会使用这个epoch的frame_idx_list
            start_batch_idx = 0 # 正常结束epoch后重置start_batch_idx
    # save final
    checkpoint_path = os.path.join(train_config['train_dir'],
        train_config['checkpoint_path'])
    config_path = os.path.join(train_config['train_dir'],
        train_config['config_path'])
    train_config_path = os.path.join(train_config['train_dir'],
        'train_config')
    print("datetime is: {}, and timestamp is: {}" \
          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), time.time()))
    print('save final checkpoint to %s' % checkpoint_path)
    saver.save(sess, checkpoint_path,
        latest_filename='checkpoint',
        global_step=results['step'])
    save_config(config_path, config_complete)
    save_train_config(train_config_path, train_config)
