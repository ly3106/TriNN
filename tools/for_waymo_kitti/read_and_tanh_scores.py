import sys
import os
import math
from waymo_open_dataset.protos import metrics_pb2
from tqdm import tqdm


def read_and_modify_scores(file_path, output_path=None):
    # 创建一个Objects实例，用于存储反序列化的数据。
    objects = metrics_pb2.Objects()

    # 从文件中读取序列化的对象。
    with open(file_path, 'rb') as f:
        objects.ParseFromString(f.read())

    # 如果没有提供输出路径，自动生成一个输出文件名。
    if not output_path:
        directory = os.path.dirname(file_path)
        filename = "tanh_scores_" + os.path.basename(file_path)
        output_path = os.path.join(directory, filename)

    # 确保输出文件的目录存在。
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 使用tqdm显示处理进度。
    for obj in tqdm(objects.objects, desc="Processing", unit="object"):
        # 使用math.tanh处理分数。
        obj.score = math.tanh(obj.score)

    # 将修改后的对象序列化并保存到新文件。
    with open(output_path, 'wb') as f:
        f.write(objects.SerializeToString())

    print(f"Processed file saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file_path> [output_file_path]")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2] if len(sys.argv) > 2 else None

    read_and_modify_scores(input_file_path, output_file_path)
