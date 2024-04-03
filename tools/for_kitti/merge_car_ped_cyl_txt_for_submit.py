import os

def merge_folders(folder1, folder2, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over files in the first folder
    for filename in os.listdir(folder1):
        if filename in os.listdir(folder2):
            with open(os.path.join(folder1, filename), 'r') as file1, open(os.path.join(folder2, filename), 'r') as file2:
                content1 = file1.read()
                content2 = file2.read()

            # Replace "-1 -1" with "0.00 0" in both contents
            merged_content = content1 + '\n' + content2

            # lines = []
            # for line in merged_content.splitlines():
            #     line_parts = line.strip().split()
            #     line_parts = line_parts[:-1]  # Remove the last number
            #     line_parts = ' '.join(line_parts)
            #     line_parts = line_parts + ' '
            #     lines.append(line_parts)
            #
            # merged_content = '\n'.join(lines)

            # Remove consecutive blank lines
            merged_content = '\n'.join(line for line in merged_content.splitlines() if line.strip())

            # Write the merged content to the output file
            output_filepath = os.path.join(output_folder, filename)
            with open(output_filepath, 'w') as output_file:
                output_file.write(merged_content)

if __name__ == "__main__":
    folder1_path = "/media/bit202/8EB8D6CCB8D6B247/Users/Public/Documents/TriNN/checkpoints/car_auto_T4_rnn_del_focal_l1l2_trainval/data"
    folder2_path = "/media/bit202/8EB8D6CCB8D6B247/Users/Public/Documents/TriNN/checkpoints/ped_cyl_auto_T4_rnn_del_focal_l1l2_trainval/data"
    output_folder_path = "/home/bit2/桌面/filtered_merge"

    merge_folders(folder1_path, folder2_path, output_folder_path)

    print("Complete")
