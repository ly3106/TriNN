#########################################################################################
# 本程序是机械扫描激光雷达深度带三角剖分算法的2D变通方案
# 程序运行的步骤为：
# 1. 生成2D点云
# 2. 复制theta左侧点云到右侧，右侧点云到左侧，完成周期闭合
# 3. 在z的上下限的上下生成每隔一定delta_theta的两行Steiner点，防止z轴上下凸包行成跨大角度的连线
#    这个连线在反投影到3D圆柱面时会形成圆柱上下面，
#    delta_theta的值最好≤激光雷达水平角度分辨率，
#    theta左右没有生成Steiner点是因为后续左右会有一部分三角形被删除，从而没必要增加Steiner点
# 4. 执行二维Delaunay三角剖分
#    相对于直接在圆柱面上进行`create_from_point_cloud_ball_pivoting`，要快大于一倍
# 5. 移除Steiner点相连的三角形
# 6. 移除复制边界附近的点后的整体点云更小的一部分边界点形成的三角形
#    这一步是为了消除delta轴左右凸包线在映射回圆柱面后形成的不合理的或交叉的三角剖分（防止非流形出现）
# 7. 将剩余复制的点的三角形的顶点索引，映射回原始点集，以进行下一步
# 8. 移除重复的三角形
# 9. 映射回圆柱面
# 10. 映射回3D空间(对于激光雷达点云来说)
##########################################################################################
# 更新日志：
# 由 cylinder_delaunay_v8_delete_slow_comment.py复制而来
##########################################################################################

import numpy as np
from scipy.spatial import Delaunay
# import matplotlib.pyplot as plt
import time

def range_belt_2d_delaunay(theta, z, steiner_theta_delta, theta_extend_width=0., theta_prune_proportion=0.,
                           steiner_theta_offset=1., steiner_z_offset=1.):

    # 周期性复制边界附近的点
    near_zero_indices = np.where(theta < theta.min() + theta_extend_width)[0]
    near_two_pi_indices = np.where(theta > theta.max() - theta_extend_width)[0]
    theta_extended = np.concatenate(
        (theta, theta[near_zero_indices] + 2 * np.pi, theta[near_two_pi_indices] - 2 * np.pi))
    z_extended = np.concatenate((z, z[near_zero_indices], z[near_two_pi_indices]))
    points_2d_extended = np.column_stack((theta_extended, z_extended))

    # 建立映射：将points_2d_extended中的点索引映射到points_2d中的点索引
    num_points = len(theta)
    # 预先计算最终的index_mapping数组的大小
    total_size = num_points + len(near_zero_indices) + len(near_two_pi_indices)
    # 初始化index_mapping为合适的大小
    index_mapping = np.empty(total_size, dtype=int)
    # 设置初始部分
    index_mapping[:num_points] = np.arange(num_points)
    # 对于near_zero_indices和near_two_pi_indices，直接在相应位置上设置值
    offset = num_points
    index_mapping[offset:offset + len(near_zero_indices)] = near_zero_indices
    offset += len(near_zero_indices)
    index_mapping[offset:offset + len(near_two_pi_indices)] = near_two_pi_indices

    # 添加Steiner点
    min_theta_extended, max_theta_extended = min(theta_extended), max(theta_extended)
    min_z, max_z = min(z_extended), max(z_extended)
    # 生成theta值
    steiner_theta = np.arange(min_theta_extended - steiner_theta_offset,
                              max_theta_extended + steiner_theta_delta + steiner_theta_offset,
                              steiner_theta_delta)  # 包括max_theta_extended
    # 生成z值
    steiner_z = np.array([min_z - steiner_z_offset, max_z + steiner_z_offset])
    ## 生成点云，每个theta值对应两个z值
    # 使用meshgrid生成点云
    theta_grid, z_grid = np.meshgrid(steiner_theta, steiner_z, indexing='ij')
    steiner_points = np.vstack([theta_grid.ravel(), z_grid.ravel()]).T
    # 将Steiner点加入点集
    points_2d_extended_with_steiner = np.vstack([np.column_stack((theta_extended, z_extended)), steiner_points])

    # 执行二维Delaunay三角剖分
    tri_with_steiner = Delaunay(points_2d_extended_with_steiner)
    # 绘制包含Steiner点的三角剖分效果
    # fig, ax = plt.subplots()
    # ax.triplot(points_2d_extended_with_steiner[:, 0], points_2d_extended_with_steiner[:, 1],
    #            tri_with_steiner.simplices.copy(), linewidth=0.125)
    # ax.plot(points_2d_extended_with_steiner[:, 0], points_2d_extended_with_steiner[:, 1], 'o', markersize=0.125)
    # ax.set_xlabel('Azimuth (rad)')
    # ax.set_ylabel('Elevation (rad)')
    # plt.tight_layout()  # 应用紧凑布局
    # plt.show()

    # 移除Steiner点相连的三角形
    triangles = tri_with_steiner.simplices.copy()
    steiner_start_index = len(theta_extended)
    # 创建一个布尔数组来标识每个三角形是否需要被保留
    # 使用np.isin检查每个三角形的顶点是否在Steiner点的索引范围内
    mask = ~np.any(np.isin(triangles, np.arange(steiner_start_index, len(points_2d_extended_with_steiner))), axis=1)
    # 使用这个mask来过滤三角形
    triangles = triangles[mask]
    # # 绘制不包含Steiner点的三角剖分效果
    # fig, ax = plt.subplots()
    # ax.triplot(points_2d_extended_with_steiner[:, 0], points_2d_extended_with_steiner[:, 1], triangles)
    # ax.plot(points_2d_extended_with_steiner[:, 0], points_2d_extended_with_steiner[:, 1], 'o', markersize=2)
    # plt.show()

    # 移除复制边界附近的点后的整体点云更小的一部分边界点形成的三角形
    boundary_value = theta_extend_width * theta_prune_proportion
    left_boundary_value = min_theta_extended + boundary_value
    right_boundary_value = max_theta_extended - boundary_value
    left_boundary_indices = np.where(theta_extended < left_boundary_value)[0]
    right_boundary_indices = np.where(theta_extended > right_boundary_value)[0]
    boundary_indices = np.concatenate((left_boundary_indices, right_boundary_indices))
    # 检查三角形中的每个点是否在边界索引中
    is_boundary_triangle = ~np.any(np.isin(triangles, boundary_indices), axis=1)
    # 使用反向布尔数组过滤三角形
    triangles = triangles[is_boundary_triangle]
    # fig, ax = plt.subplots()
    # ax.triplot(points_2d_extended_with_steiner[:, 0], points_2d_extended_with_steiner[:, 1], triangles)
    # ax.plot(points_2d_extended_with_steiner[:, 0], points_2d_extended_with_steiner[:, 1], 'o', markersize=2)
    # plt.show()

    # 更新三角形的顶点索引，映射回原始点集
    mapped_triangles = index_mapping[triangles]
    # 对每个三角形的顶点进行排序
    sorted_triangles = np.sort(mapped_triangles, axis=1)
    # 使用np.unique去除重复的三角形，返回索引
    _, unique_indices = np.unique(sorted_triangles, axis=0, return_index=True)
    # 使用这些索引获取唯一的三角形
    unique_triangles = mapped_triangles[unique_indices]

    return unique_triangles



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 参数
    num_points = 200  # 生成的点数
    cylinder_height = 5  # 圆柱高度
    cylinder_radius = 1  # 圆柱半径

    # 在圆柱面上生成随机点
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(0, cylinder_height, num_points)

    unique_triangles = range_belt_2d_delaunay(theta, z, 0.2, np.pi / 4., 0.5)

    fig, ax = plt.subplots()
    ax.triplot(theta, z, unique_triangles)
    ax.plot(theta, z, 'o', markersize=2)
    plt.show()

    # 映射回圆柱面
    x = cylinder_radius * np.cos(theta)
    y = cylinder_radius * np.sin(theta)
    # 绘制最终结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z, triangles=unique_triangles, cmap=plt.cm.Spectral)
    ax.scatter(x, y, z, s=4)
    # 设置图表
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()