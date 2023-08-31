# 从文件夹A中选出文件夹B中相同文件名（不包括扩展名）的文件移动到一个新的文件夹中


import os
import shutil

# 定义文件夹路径
folder_a_path = '/media/bit202/680CCE330CCDFBD6/KITTI/object/devkit_object/matlab/jpgs'
folder_b_path = '/media/bit202/680CCE330CCDFBD6/KITTI/object/labels/training/label_2_合并_0.6.4_0.6.5'
output_folder_path = '/media/bit202/680CCE330CCDFBD6/KITTI/object/devkit_object/matlab/jpgs2'

# 获取文件夹B中的文件名（不包括扩展名）
folder_b_files = [os.path.splitext(file)[0] for file in os.listdir(folder_b_path) if os.path.isfile(os.path.join(folder_b_path, file))]

# 遍历文件夹A中的文件
for file in os.listdir(folder_a_path):
    file_name, file_extension = os.path.splitext(file)
    
    # 检查文件名是否在文件夹B中存在
    if file_name in folder_b_files:
        # 构建源文件路径和目标文件路径
        source_file_path = os.path.join(folder_a_path, file)
        target_file_path = os.path.join(output_folder_path, file)
        
        # 移动文件到新的文件夹中
        shutil.move(source_file_path, target_file_path)
        
print("文件移动完成！")
