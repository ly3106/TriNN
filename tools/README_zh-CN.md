# 使用说明
以下是一些辅助操作工具。这里是它们的文件结构：

    tools
    ├─── for_kitti
    │    ├─── filter_score_car.py
    │    ├─── filter_score_ped_cyl.py
    │    ├─── merge_car_ped_cyl_txt.py
    │    ├─── merge_car_ped_cyl_txt_for_submit.py
    │    └─── move_file.py
    └─── for_waymo_kitti
         ├─── grep_car_to_txt.sh
         ├─── grep_pedestrian_cyclist_to_txt.sh
         └─── read_and_tanh_scores.py
## 1. `for_kitti`
这个目录中的工具旨在后处理`run_for_kitti.py`生成的`txt`文件。它们用于生成最终提交到KITTI官网的文件。

### 1.1 `merge_car_ped_cyl_txt_for_submit.py`
运行`run_for_kitti.py`后，你将得到两个包含最终预测结果的`txt`文件目录，分别对应car和pedestrian、cyclist。要将这些文件提交给KITTI网站，必须将它们合并到一个目录中。运行`merge_car_ped_cyl_txt_for_submit.py`来完成这项工作。

执行此脚本之前，根据你的实际路径修改其`main`部分中的`folder1_path`、`folder2_path`和`output_folder_path`。

### 1.2 可视化
如果你想可视化预测效果：

1. 筛选出Score超过特定阈值的对象：
```bash
filter_score_car.py
```
```bash
filter_score_ped_cyl.py
```
运行之前，调整这些脚本中的`input_folder`、`output_folder`和行列`9:93`处的Score阈值。

2. `merge_car_ped_cyl_txt.py`
执行此脚本将筛选后的文件合并到一个`txt`文件的目录中。与`merge_car_ped_cyl_txt_for_submit.py`不同，此脚本保留了Score值。

3. 可视化工具
可以使用各种工具来可视化这些`txt`文件，类似于可视化KITTI的标注了的数据。很多工具都可以用于此目的，比如KITTI自己的MATLAB工具。

## 2. `for_waymo_kitti`
### 2.1 `.sh`
这些bash脚本用于提取包含特定对象的WOD的`ImageSets`文件。生成的文件已经放在`splits/waymo_kitti/include_objects`中了。
### 2.2 `read_and_tanh_scores.py`
使用`tanh`函数将Score转换为[0,1]区间。详细使用方法，请参见[此处](../README.md)的“Submission”部分。"