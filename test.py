import numpy as np
import open3d as o3d

# 创建一个 LineSet 对象
lineset = o3d.geometry.LineSet()

# 添加一些线段
lineset.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
lineset.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [1, 2]]))

# 获取所有线段的起点和终点坐标，并计算每个线段的平方和
line_coords = []
length_squared_sum = []
for i in range(np.asarray(lineset.lines).shape[0]):
    start, end = lineset.get_line_coordinate(i)
    line_coords.append((start, end))
    length_squared_sum.append(np.sum(np.square(end - start)))

arr = np.array(length_squared_sum)
# 判断数组中的值是否大于 1，并返回一个 mask 布尔型数组
mask = arr < 1

print(mask)

print("Line coordinates: ")
print(line_coords)
print("Length squared sum: ", length_squared_sum)


import numpy as np
import open3d as o3d

# 创建 LineSet 对象
lineset = o3d.geometry.LineSet()
lineset.points = o3d.utility.Vector3dVector([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
lineset.lines = o3d.utility.Vector2iVector([(0, 1), (1, 2), (2, 0), (0, 2), (1, 2)])

# 计算每条边的长度
coords = np.asarray(lineset.points)
lines = np.asarray(lineset.lines)
dists = np.linalg.norm(coords[lines[:, 0]] - coords[lines[:, 1]], axis=1)

# 挑出边长小于 1 的边
mask = dists > 1
selected_lines = lines[mask]
selected_lineset = o3d.geometry.LineSet()
selected_lineset.points = lineset.points
selected_lineset.lines = o3d.utility.Vector2iVector(selected_lines)

# 打印挑出来的边
print(selected_lineset.lines)
print('selected_lines:', selected_lines)
print('dists',dists)


import open3d as o3d


# 从网格创建线段集合
line_set = lineset
lines = line_set.lines
lines = np.asarray(line_set.lines)


def check_for_duplicate_edges(lines):
    num_lines = lines.shape[0]
    for i in range(num_lines):
        current_start, current_end = lines[i]
        for j in range(i):
            prev_start, prev_end = lines[j]
            if (current_start == prev_start and current_end == prev_end) or \
                    (current_start == prev_end and current_end == prev_start):
                print(f"Duplicate edge found: ({current_start}, {current_end})")

def check_for_duplicate_edges_2(lines):
    edge_dict = {}
    for start, end in lines:
        # 确保 start < end
        if start > end:
            start, end = end, start
        if (start, end) in edge_dict:
            print(f"Duplicate edge found: ({start}, {end})")
        else:
            edge_dict[(start, end)] = True


# 检查是否存在重复边
check_for_duplicate_edges(lines)
check_for_duplicate_edges_2(lines)
