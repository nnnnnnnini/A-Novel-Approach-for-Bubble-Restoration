import os
import random
'''
# 设置目录路径
base_dir = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/results/bubble_new_add41bg_duibi/bubble_zuzhi_patch"
output_dir = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/results/bubblezuzhi_patch_name"

# 获取所有子目录
sub_dirs = [x[0] for x in os.walk(base_dir)]

# 收集所有图片文件
all_image_files = []
for dir_path in sub_dirs:
    image_files = [f for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    all_image_files.extend([(dir_path, f) for f in image_files])

# 从所有图片中随机选择100张
selected_images = random.sample(all_image_files, min(100, len(all_image_files)))

# 将选中的图片文件名加上子目录名写入txt文件中
output_txt_path = os.path.join(output_dir, "selected_images.txt")
with open(output_txt_path, 'w') as file:
    for dir_path, image_name in selected_images:
        file.write(f"{os.path.basename(dir_path)}/{image_name}\n")

print("Done!")
'''

import shutil

# 设置目录路径和txt文件路径
source_dir = "/home/xingzehang/project_100t/jiaoda1/DATA/all_bubble_patch"
output_dir = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/results/evaluate_100bubble_patch"
txt_file_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/results/bubblezuzhi_patch_name/selected_images.txt"

# 创建存放图片的输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取txt文件中的子目录名和图片名组合
with open(txt_file_path, 'r') as file:
    sub_dir_image_pairs = file.read().splitlines()

# 遍历子目录名和图片名组合列表，复制对应的图片文件到输出目录
for sub_dir_image_pair in sub_dir_image_pairs:
    sub_dir, image_name = sub_dir_image_pair.split('/')
    name = sub_dir + "_" + image_name
    source_path = os.path.join(source_dir, sub_dir, name)
    output_path = os.path.join(output_dir, name)

    if os.path.exists(source_path):
        shutil.copyfile(source_path, output_path)
        print(f"Copied {image_name} from {sub_dir} to {output_dir}")
    else:
        print(f"File {image_name} in directory {sub_dir} not found.")

print("Done!")