# Copy from TriGCN
# Update log
# 20230511 Li v1.1 Select out the edge which distance less than a value, and Add reverse edges, self loops.
# 20230512 Lℹ v1.2 Add 'visualize_graph()'
# =================================================================

from scipy.spatial import Delaunay
from collections import namedtuple  # , defaultdict

# from o3d import *
import open3d as o3d
import math
# import dgl
import tensorflow as tf
# from dataset.kitti_data_parse import parse_kitti_args
import numpy as np
# import matplotlib.pyplot as plt

Points = namedtuple('Points', ['xyz', 'attr'])

def get_velo_points(frame_idx, xyz_range=None):  # self,
    """Load velo points from frame_idx.

    Args:
        frame_idx: the index of the frame to read.

    Returns: Points.
    """

    #     point_file = join(self._point_dir, self._file_list[frame_idx])+'.bin'
    point_file = frame_idx + '.bin'
    velo_data = np.fromfile(point_file, dtype=np.float32).reshape(-1, 4)
    velo_points = velo_data[:, :3]
    reflections = velo_data[:, [3]]
    if xyz_range is not None:
        x_range, y_range, z_range = xyz_range
        mask = (
                       velo_points[:, 0] > x_range[0]) * (velo_points[:, 0] < x_range[1])
        mask *= (
                        velo_points[:, 1] > y_range[0]) * (velo_points[:, 1] < y_range[1])
        mask *= (
                        velo_points[:, 2] > z_range[0]) * (velo_points[:, 2] < z_range[1])
        return Points(xyz=velo_points[mask], attr=reflections[mask])
    return Points(xyz=velo_points, attr=reflections)


# point_cloud_xyzi = get_velo_points(frame_idx='/media/bit202/TOSHIBA EXT1/数据集/KITTI/object/velodyne/training/velodyne原文件/000200')
# xyz = point_cloud_xyzi.xyz#[0]
# intensity = point_cloud_xyzi.attr#[1]

# r = np.sqrt(pow(xyz[:,0], 2) + pow(xyz[:,1], 2))
# azimuth = np.arctan2(xyz[:,1],  xyz[:,0])
# elevation = np.arctan2(xyz[:,2], r)
# azimuth_resolution_rad = 0.09*math.pi / 180
# azimuth_norm = azimuth / azimuth_resolution_rad
# elevation_resolution_rad = 26.8*math.pi/180/(64-1)
# elevation_norm = elevation / elevation_resolution_rad



def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range



# print('Draw intensity image')
# intensity_colors = normalization(intensity[:,0])
# plt.scatter(-azimuth_norm, elevation_norm, s=0.5, c=intensity_colors,  cmap='rainbow', alpha=0.5)
# plt.show()
#
# print('Draw range image')
# r_norm=normalization(r)
# # plt.scatter(-azimuth, elevation, s=0.5, c=r_norm,  cmap='gist_ncar', alpha=0.5) #因方位角正方向与笛卡尔坐标系左右方向相反故增加负号#Since the positive direction of the azimuth angle is opposite to the left and right directions of the Cartesian coordinate system, a negative sign is added
# plt.scatter(-azimuth_norm, elevation_norm, s=0.5, c=r_norm,  cmap='gist_ncar', alpha=0.5)
# plt.show()
#
# print('Delaunay triangulation of range image and show it')
# from scipy.spatial import Delaunay
# tri = Delaunay(np.dstack((-azimuth, elevation))[0,:,:])
# plt.triplot(-azimuth, elevation, tri.simplices)
# plt.show()
#
# print('Project 2D delaunay triangulation to 3D')
# origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(xyz)
# print("Construct mesh in Open3D...")
# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = point_cloud.points
# mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tri.simplices))
# print("Computing normal and rendering it.")
# mesh.compute_vertex_normals()
# mesh.paint_uniform_color([1, 0.706, 0])
#
# print("Range loop projection")
# xoy_r_unit = 0.5 / np.sin(azimuth_resolution_rad)
# zoom_rate = r / xoy_r_unit
# xyz2 = xyz / np.dstack((zoom_rate, zoom_rate, zoom_rate))[0,:,:]
# xyz2[:,2] = elevation_norm
# range_loop_point_cloud = o3d.geometry.PointCloud()
# range_loop_point_cloud.points = o3d.utility.Vector3dVector(xyz2)
# colors = plt.get_cmap("gist_ncar")(r_norm)
# colors[r_norm < 0] = 0
# range_loop_point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([origin_frame, range_loop_point_cloud, point_cloud, mesh])
#
# print("Recompute the normal of the range loop point cloud")
# range_loop_point_cloud.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=100))
# range_loop_point_cloud.orient_normals_towards_camera_location()
# o3d.visualization.draw_geometries([range_loop_point_cloud])#,
# print("Compute TriangleMesh with ball pivoting")
# # radii = [0.005, 0.01, 0.02, 0.04]
# # radii = [1., 2., 3., 4.]
# radii = [0.5, 1., 2., 8.]
# range_loop_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     range_loop_point_cloud, o3d.utility.DoubleVector(radii))
# range_loop_mesh.paint_uniform_color([1, 0.706, 0])
# # o3d.visualization.draw_geometries([range_loop_mesh])
# range_loop_line_set = o3d.geometry.LineSet.create_from_triangle_mesh(range_loop_mesh)
# colors = [[1, 0, 0] for i in range(len(range_loop_line_set.lines))]
# range_loop_line_set.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([origin_frame, range_loop_point_cloud, range_loop_line_set])#, zoom=0.8
#
# # range_loop_mesh.vertices = point_cloud.points
# range_loop_line_set.points = point_cloud.points
# o3d.visualization.draw_geometries([origin_frame, point_cloud, range_loop_line_set])



def generate_range_image_xy(points_xyz):
    '''
    Args:
    --------
    3D point cloud including (x, y, z)

    Returns:
    --------
    2D range image coordinates including (-azimuth, elevation)
    '''
    xoy_r = np.sqrt(pow(points_xyz[:, 0], 2) + pow(points_xyz[:, 1], 2))
    azimuth = np.arctan2(points_xyz[:, 1], points_xyz[:, 0])
    elevation = np.arctan2(points_xyz[:, 2], xoy_r)

    '''是否需要归一化回头再讨论'''
    # azimuth_resolution_rad = 0.09*math.pi / 180
    # azimuth_norm = azimuth / azimuth_resolution_rad
    # elevation_resolution_rad = 26.8*math.pi/180/(64-1)
    # elevation_norm = elevation / elevation_resolution_rad

    '''这个拼接方式效率如何再做讨论'''
    range_image_xy = np.dstack((-azimuth, elevation))[0, :, :]
    return range_image_xy


def generate_line_set_with_2D_delaunay(points_xyz, xy):
    ''' 使用2D Delaunay三角剖分生成图拓扑结构
    
    Args:
    points_xyz: a numpy array[n, 3], including n raw 3D points,and each point includes (x, y, z);
                        一个[n, 3]的numpy数组，其包括n个3D点，每个3D点包含(x, y, z) 3个坐标元素；

    xy: a numpy array [n, 2] including n 2D points projected by the points_xyz, and each point includes (x, y).
        一个[n, 2]的numpy数组，其包含n个2D点，这些2D点是points_xyz经过某种投影得来的，每个点包含(x, y) 2个坐标元素。

    Return: a o3d.geometry.LineSet including [n, 3] points and [m, 2] lines. There are total m lines, and each lines included by 2 ints that indicate the index of points
                    一个o3d.geometry.LineSet对象，其由 [n, 3]个点和[m, 2]个线对组成。共有m个线对，每个线对由两个整数组成，这个整数代表对应点的下标索引
    '''
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)

    tri = Delaunay(xy)

    triangle_mesh = o3d.geometry.TriangleMesh()
    triangle_mesh.vertices = point_cloud.points
    triangle_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tri.simplices))
    triangle_line_set = o3d.geometry.LineSet.create_from_triangle_mesh(triangle_mesh)
    # triangle_line_set.points = point_cloud.points
    return triangle_line_set


'''
def generate_graph_with_triangle_line_set(triangle_line_set):
    edges = np.asarray(triangle_line_set.lines)
    edges = tf.constant(edges[:, 0]), tf.constant(edges[:, 1])
    out_graph = dgl.graph(edges)
    # out_graph = dgl.graph(edges).to('gpu:0')
    return out_graph
'''


def graph_add_features(in_graph, cls_labels, vertexes_xyz, vertexes_attr):
    out_graph = in_graph
    out_graph.ndata['xyz'] = tf.constant(vertexes_xyz)
    # out_graph.ndata['i'] = tf.constant(intensity)
    out_graph.ndata['i'] = tf.constant(vertexes_attr[:, [0]])

    # 先不进行图边属性计算
    # delta_xyz = []
    # delta_distance = []
    # for line in np.asarray(triangle_line_set.lines):
    #     delta_xyz.append((xyz[line[1], :] - xyz[line[0], :]).tolist())
    # delta_xyz = np.array(delta_xyz)
    # delta_distance = np.sqrt(
    #     pow(delta_xyz[:, 0], 2) +
    #     pow(delta_xyz[:, 1], 2) +
    #     pow(delta_xyz[:, 2], 2)
    # )
    # out_graph.edata['delta_xyz'] = tf.constant(delta_xyz)
    # out_graph.edata['delta_distance'] = tf.constant(delta_distance)
    # out_graph = out_graph.to('gpu:0')
    # out_graph.edata['delta_xyz'] = tf.ones(out_graph.num_edges(), 3)
    # out_graph.ndata['feature'] = tf.ones(out_graph.num_nodes(), 3)

    out_graph.ndata['feat'] = out_graph.ndata['i']
    out_graph.ndata['label'] = tf.constant(cls_labels)
    out_graph.ndata['train_mask'] = tf.constant([True for i in range(0, int(out_graph.num_nodes() * 0.5))] +
                                                [False for i in
                                                 range(int(out_graph.num_nodes() * 0.5), out_graph.num_nodes())])
    out_graph.ndata['val_mask'] = tf.constant([False for i in range(int(out_graph.num_nodes() * 0.5))] +
                                              [True for i in range(int(out_graph.num_nodes() * 0.5),
                                                                   int(out_graph.num_nodes() * 0.8))] +
                                              [False for i in
                                               range(int(out_graph.num_nodes() * 0.8), out_graph.num_nodes())])
    out_graph.ndata['test_mask'] = tf.constant([False for i in range(int(out_graph.num_nodes() * 0.8))] +
                                               [True for i in
                                                range(int(out_graph.num_nodes() * 0.8), int(out_graph.num_nodes()))])
    return out_graph


def check_for_duplicate_edges(lines):
    '''（中）使用哈希表来存储边的起点和终点，这个方法的时间复杂度是线性的，与边的数量成正比。

    (EN) Using a hash table to store the start and end points of edges, the time complexity of this method is linear, proportional to the number of edges.'''
    edge_dict = {}
    for start, end in lines:
        # 确保 start < end
        if start > end:
            start, end = end, start
        if (start, end) in edge_dict:
            print(f"Duplicate edge found: ({start}, {end})")
        else:
            edge_dict[(start, end)] = True


def generate_graph_edges(
        points_xyz: np.array, max_distance=1.,
        make_undirected=False, add_self_loops=False) -> np.array:  # TODO: 这个值是超参数需要调节
    xy = generate_range_image_xy(points_xyz)  # - [0., 0., 100.]
    line_set = generate_line_set_with_2D_delaunay(points_xyz, xy)
    lines = np.asarray(line_set.lines)
    # check_for_duplicate_edges(lines)

    # 计算每条边的长度
    coords = np.asarray(line_set.points)
    dists = np.linalg.norm(coords[lines[:, 0]] - coords[lines[:, 1]], axis=1)

    # （中）挑选出边长小于最大距离上限的边
    # (En) Select the edges whose distance is less than the max distance (i.e. the upper limit of the distance)
    mask = dists < max_distance
    selected_lines = lines[mask]

    # （中）添加反向边，使其成为无向图
    # (EN) Add reverse edges, make it to undirected graph
    if make_undirected:
        selected_lines = np.vstack([selected_lines, selected_lines[:, ::-1]])

    # （中）添加自环
    # (EN) Add self loop
    # all_nodes = np.unique(selected_lines)
    if add_self_loops:
        all_nodes = np.arange(coords.shape[0])
        self_connections = np.stack([all_nodes, all_nodes], axis=1)
        selected_lines = np.vstack([selected_lines, self_connections])

    return selected_lines  # lines


def generate_dgl_graph(points_xyz):
    xy = generate_range_image_xy(points_xyz)
    triangle_line_set = generate_line_set_with_2D_delaunay(points_xyz, xy)
    out_graph = generate_graph_with_triangle_line_set(triangle_line_set)
    # out_graph = graph_add_features(out_graph, cls_labels, points_xyzi)
    return out_graph


# def generate_graph_with_range_image_test():
#     '''
#     使用深度图构造图
#     '''
#     point_cloud_xyzi = get_velo_points(frame_idx='/media/bit202/680CCE330CCDFBD6/KITTI/object/velodyne/training/velodyne/000200')
#
#     xyz = point_cloud_xyzi.xyz
#     intensity = point_cloud_xyzi.attr
#
#     cam_rgb_points, cls_labels, label_map, normals, lower, upper = fetch_data(frame_idx = 110)
#     projected = np.matmul(cam_rgb_points.xyz, np.transpose(normals))
#
#     xyz = cam_rgb_points.xyz
#     xoy_r = np.sqrt(pow(xyz[:,0], 2) + pow(xyz[:,1], 2))
#     azimuth = np.arctan2(xyz[:,1],  xyz[:,0])
#     elevation = np.arctan2(xyz[:,2], xoy_r)
#     azimuth_resolution_rad = 0.09*math.pi / 180
#     azimuth_norm = azimuth / azimuth_resolution_rad
#     elevation_resolution_rad = 26.8*math.pi/180/(64-1)
#     elevation_norm = elevation / elevation_resolution_rad
#
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(xyz)
#
#     from scipy.spatial import Delaunay
#     tri = Delaunay(np.dstack((-azimuth, elevation))[0,:,:])
#
#     range_image_mesh = o3d.geometry.TriangleMesh()
#     range_image_mesh.vertices = point_cloud.points
#     range_image_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tri.simplices))
#
#     range_image_line_set = o3d.geometry.LineSet.create_from_triangle_mesh(range_image_mesh)
#     range_image_line_set.points = point_cloud.points
#
#     edges = np.asarray(range_image_line_set.lines)
#     edges = tf.constant(edges[:, 0]), tf.constant(edges[:, 1])
#     out_graph = dgl.graph(edges)
#     # out_graph = dgl.graph(edges).to('gpu:0')
#
#     out_graph.ndata['xyz'] = tf.constant(xyz)
#     # out_graph.ndata['i'] = tf.constant(intensity)
#     out_graph.ndata['i'] = tf.constant(cam_rgb_points.attr[:, [0]])
#
#     delta_xyz = []
#     delta_distance = []
#     for line in np.asarray(range_image_line_set.lines):
#         delta_xyz.append((xyz[line[1], :] - xyz[line[0], :]).tolist())
#     delta_xyz = np.array(delta_xyz)
#     delta_distance = np.sqrt(
#         pow(delta_xyz[:, 0], 2) +
#         pow(delta_xyz[:, 1], 2) +
#         pow(delta_xyz[:, 2], 2)
#     )
#     out_graph.edata['delta_xyz'] = tf.constant(delta_xyz)
#     out_graph.edata['delta_distance'] = tf.constant(delta_distance)
#     # out_graph = out_graph.to('gpu:0')
#     # out_graph.edata['delta_xyz'] = tf.ones(out_graph.num_edges(), 3)
#     # out_graph.ndata['feature'] = tf.ones(out_graph.num_nodes(), 3)
#
#     out_graph.ndata['feat'] = out_graph.ndata['i']
#     from random import choice
#     out_graph.ndata['label'] = tf.constant(cls_labels)#[choice([1, 2]) for i in range(out_graph.num_nodes())], dtype = tf.int64)
#     out_graph.ndata['train_mask'] = tf.constant([True for i in range(0, int(out_graph.num_nodes()*0.5))] +
#                                                                                                 [False for i in range(int(out_graph.num_nodes()*0.5), out_graph.num_nodes())])
#     out_graph.ndata['val_mask'] = tf.constant([False for i in range(int(out_graph.num_nodes()*0.5))] +
#                                                                                             [True for i in range(int(out_graph.num_nodes()*0.5), int(out_graph.num_nodes()*0.8))] +
#                                                                                             [False for i in range(int(out_graph.num_nodes()*0.8), out_graph.num_nodes())])
#     out_graph.ndata['test_mask'] = tf.constant([False for i in range(int(out_graph.num_nodes()*0.8))] +
#                                                                                               [True for i in range(int(out_graph.num_nodes()*0.8), int(out_graph.num_nodes()))])
#     return out_graph


# loop深度环

def generate_graph_with_range_belt():  # 深度带
    point_cloud_xyzi = get_velo_points(
        frame_idx='/media/bit202/680CCE330CCDFBD6/KITTI/object/velodyne/training/velodyne原文件/000200')
    xyz = point_cloud_xyzi.xyz
    intensity = point_cloud_xyzi.attr

    xoy_r = np.sqrt(pow(xyz[:, 0], 2) + pow(xyz[:, 1], 2))
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    elevation = np.arctan2(xyz[:, 2], xoy_r)
    azimuth_resolution_rad = 0.09 * math.pi / 180
    azimuth_norm = azimuth / azimuth_resolution_rad
    elevation_resolution_rad = 26.8 * math.pi / 180 / (64 - 1)
    elevation_norm = elevation / elevation_resolution_rad

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    print("Range loop projection")
    xoy_r_unit = 0.5 / np.sin(azimuth_resolution_rad)  # 单位化半径，使水平相邻两点理论直线距离为1
    zoom_rate = xoy_r / xoy_r_unit
    xyz2 = xyz / np.dstack((zoom_rate, zoom_rate, zoom_rate))[0, :, :]
    xyz2[:, 2] = elevation_norm
    range_loop_point_cloud = o3d.geometry.PointCloud()
    range_loop_point_cloud.points = o3d.utility.Vector3dVector(xyz2)

    print("Recompute the normal of the range loop point cloud")
    range_loop_point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=100))
    range_loop_point_cloud.orient_normals_towards_camera_location()
    print("Compute TriangleMesh with ball pivoting")
    radii = [0.5, 1., 2., 8.]
    range_loop_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        range_loop_point_cloud, o3d.utility.DoubleVector(radii))
    range_loop_line_set = o3d.geometry.LineSet.create_from_triangle_mesh(range_loop_mesh)
    range_loop_line_set.points = point_cloud.points

    edges = np.asarray(range_loop_line_set.lines)
    # edges = np.asarray(lines)
    edges = tf.constant(edges[:, 0]), tf.constant(edges[:, 1])
    out_graph = dgl.graph(edges)

    # Calculate features
    # out_graph.ndata['x'] = tf.constant(xyz[:, 0])
    # out_graph.ndata['y'] = tf.constant(xyz[:, 1])
    # out_graph.ndata['z'] = tf.constant(xyz[:, 2])
    # out_graph.ndata['i'] = tf.constant(intensity)
    # out_graph.ndata['feature'] = tf.ones(out_graph.num_nodes(), 3)    

    return out_graph


def generate_graph_with_range_disk():  # 深度盘
    pass


def generate_graph_with_range_sphere():  # 深度球
    pass


def visualize_graph():
    '''
    使用深度图构造图
    '''

    # 获取原始点云
    point_cloud_xyzi = get_velo_points(frame_idx='/media/bit202/680CCE330CCDFBD6/KITTI/object/velodyne/training/velodyne/000200')

    xyz = point_cloud_xyzi.xyz
    intensity = point_cloud_xyzi.attr

    # cam_rgb_points, cls_labels, label_map, normals, lower, upper = fetch_data(frame_idx = 110)
    # projected = np.matmul(cam_rgb_points.xyz, np.transpose(normals))
    #
    # xyz = cam_rgb_points.xyz

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    edges = generate_graph_edges(xyz, make_undirected=True, add_self_loops=True)

    line_set = o3d.geometry.LineSet(
        points=point_cloud.points,
        lines=o3d.utility.Vector2iVector(edges),
    )
    colors = [[1, 0, 0] for i in range(len(line_set.lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([line_set])

    return

if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    # out_graph = generate_graph_with_range_belt(range_loop_line_set.lines)
    # out_graph = generate_graph_with_range_belt()
    # out_graph = generate_graph_with_range_image_test()
    out_graph = visualize_graph()  # TODO: 能运行在tf2.4下不能运行在2070 tf1.15.0下，需要修改
