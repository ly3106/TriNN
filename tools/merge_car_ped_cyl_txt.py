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
            merged_content = content1.replace("-1 -1", "0.00 0") + '\n' + content2.replace("-1 -1", "0.00 0")

            lines = []
            for line in merged_content.splitlines():
                line_parts = line.strip().split()
                line_parts = line_parts[:-1]  # Remove the last number
                line_parts = ' '.join(line_parts)
                line_parts = line_parts + ' '
                lines.append(line_parts)

            merged_content = '\n'.join(lines)

            # Remove consecutive blank lines
            merged_content = '\n'.join(line for line in merged_content.splitlines() if line.strip())

            # Write the merged content to the output file
            output_filepath = os.path.join(output_folder, filename)
            with open(output_filepath, 'w') as output_file:
                output_file.write(merged_content)

if __name__ == "__main__":
    folder1_path = "/home/bit202/桌面/filtered_data_0.6.4"
    folder2_path = "/home/bit202/桌面/filtered_data_0.6.5"
    output_folder_path = "/home/bit202/桌面/filtered_合并"

    merge_folders(folder1_path, folder2_path, output_folder_path)

    print("Complete")
