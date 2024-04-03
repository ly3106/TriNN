import os

def process_file(input_path, output_path):
    with open(input_path, 'r') as input_file:
        lines = input_file.readlines()

    filtered_lines = [line for line in lines if line.strip() and float(line.split()[-1]) >= 6.59]

    with open(output_path, 'w') as output_file:
        output_file.writelines(filtered_lines)

def process_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹（如果不存在）

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            process_file(input_path, output_path)
            print(f"Processed: {filename}")

if __name__ == "__main__":
    input_folder = "/home/bit202/桌面/data_0.6.4"  # 输入文件夹路径
    output_folder = "/home/bit202/桌面/filtered_data_0.6.4"  # 输出文件夹路径
    process_folder(input_folder, output_folder)
    print("Task completed.")
