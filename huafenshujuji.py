import os
import shutil
import random
bubble_path = "/home/xingzehang/project_100t/jiaoda1/0429nengyong/20240412fyy_20x_512"
qietu_bubble_path = "/home/xingzehang/project_100t/jiaoda1/0429nengyong/test_nengyong_20x_512_suoluetu_sf_bgAt01_0507/patch"
qietu_normal_path = "/home/xingzehang/project_100t/jiaoda1/0429nengyong/train_no_pair_20x_512_suoluetu_sf_bgAt1_0507/patch"
target_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/datasets/bubble0412_41"
os.makedirs(target_path, exist_ok=True)
target_path_trainA = os.path.join(target_path,"trainA")
os.makedirs(target_path_trainA, exist_ok=True)
target_path_trainB = os.path.join(target_path,"trainB")
os.makedirs(target_path_trainB, exist_ok=True)
# 处理 bubble_path 中的图片
for folder_name in os.listdir(bubble_path):
    folder_path = os.path.join(bubble_path, folder_name)
    target_folder_path = os.path.join(target_path_trainA, folder_name, "tissue")
    os.makedirs(target_folder_path, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            shutil.copy(os.path.join(folder_path, file_name), target_folder_path)

    # 处理 qietu_bubble_path 中的图片
    qietu_folder_path = os.path.join(qietu_bubble_path, folder_name, "bg")
    bg_target_folder_path = os.path.join(target_path_trainA, folder_name, "bg")
    os.makedirs(bg_target_folder_path, exist_ok=True)

    files = os.listdir(qietu_folder_path)
    num_files_to_copy = int(len(os.listdir(folder_path)) * 0.25)
    random.shuffle(files)  # 打乱文件列表顺序
    files_to_copy = files[:num_files_to_copy]

    for file_name in files_to_copy:
        # a=os.path.join(qietu_folder_path, file_name)
        # pass
        shutil.copy(os.path.join(qietu_folder_path, file_name), bg_target_folder_path)

#处理 qietu_normal_path 中的图片
for folder_name in os.listdir(qietu_normal_path):
    bg_folder_path = os.path.join(qietu_normal_path, folder_name, "bg")
    bg_target_folder_path = os.path.join(target_path_trainB, folder_name, "bg")
    os.makedirs(bg_target_folder_path, exist_ok=True)

    bg_files = os.listdir(bg_folder_path)
    random.shuffle(bg_files)
    bg_files_to_copy = bg_files[:40]  # 只取前40张图片

    for file_name in bg_files_to_copy:
        # a = os.path.join(bg_folder_path, file_name)
        #pass
        shutil.copy(os.path.join(bg_folder_path, file_name), bg_target_folder_path)
for folder_name in os.listdir(qietu_normal_path):
    bg_folder_path = os.path.join(qietu_normal_path, folder_name, "tissue")
    bg_target_folder_path = os.path.join(target_path_trainB, folder_name, "tissue")
    os.makedirs(bg_target_folder_path, exist_ok=True)

    bg_files = os.listdir(bg_folder_path)
    random.shuffle(bg_files)
    bg_files_to_copy = bg_files[:160]  # 只取前160张图片

    for file_name in bg_files_to_copy:
        # a = os.path.join(bg_folder_path, file_name)
        # pass
        shutil.copy(os.path.join(bg_folder_path, file_name), bg_target_folder_path)

