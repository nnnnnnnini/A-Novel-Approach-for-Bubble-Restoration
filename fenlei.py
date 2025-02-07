import os
import shutil

# 源文件夹路径
source_folder = '/home/xingzehang/project_100t/tuisexiufu_results/ffpep_cyclegan/test_50/images/fake_B'
output_folder = '/home/xingzehang/project_100t/tuisexiufu_results/ffpep_cyclegan/test_50/fakeB_fenlei'
# 获取源文件夹下的所有文件列表
files = os.listdir(source_folder)

# 创建目标文件夹
for file_name in files:
    # 获取文件夹名
    #folder_letter = file_name.split("_")[0]
    parts = file_name.split(".")[0].split("_")
    folder_letter = "_".join(parts[:len(parts)-2])
    # 构建目标文件夹的路径
    target_folder = os.path.join(output_folder, folder_letter)

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 构建源文件的完整路径
    source_file_path = os.path.join(source_folder, file_name)

    # 构建目标文件的完整路径
    target_file_path = os.path.join(target_folder, file_name)
    # 将文件移动到目标文件夹
    shutil.move(source_file_path, target_file_path)

print("文件已按名称分别存入不同的文件夹。")
