import glob
import shutil

from PIL import Image
import os
import cv2
# folder_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/datasets/bubble0412/trainA"
# target_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/datasets/bubble0412/trainA_tianchong"
folder_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/datasets/bubble0412/trainA_tianchong"
for subdir in os.listdir(folder_path):
    subdir_path = os.path.join(folder_path,subdir)
    bg_subdir_path = os.path.join(subdir_path,"tissue")
    patch_size = (512,512)
    for file_name in os.listdir(bg_subdir_path):
        try:
            # 使用 OpenCV 打开图像
            image = cv2.imread(os.path.join(bg_subdir_path,file_name))

            # 检查图像大小是否符合要求
            if image.shape[1] != patch_size[1] or image.shape[0] != patch_size[0]:
                print("tianchong",file_name)
        except Exception as e:
                # 捕获异常
                print(f"处理文件 {file_name} 时出错: {e}")