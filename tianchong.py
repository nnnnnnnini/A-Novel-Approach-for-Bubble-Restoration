import glob
import shutil

from PIL import Image
import os
import cv2
# folder_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/datasets/bubble0412/trainA"
# target_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/datasets/bubble0412/trainA_tianchong"
# folder_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/datasets/bubble0412/trainB"
# target_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/datasets/bubble0412/trainB_tianchong"
folder_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/datasets/bubble0412_41/trainB"
target_path = "/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/datasets/bubble0412_41/trainB_tianchong"
os.makedirs(target_path,exist_ok=True)
for subdir in os.listdir(folder_path):
    subdir_path = os.path.join(folder_path,subdir)
    target_subdir_path = os.path.join(target_path,subdir)
    bg_subdir_path = os.path.join(subdir_path,"tissue")
    target_subdir_path = os.path.join(target_subdir_path,"tissue")
    os.makedirs(target_subdir_path,exist_ok=True)
    patch_size = (512,512)
    for file_name in os.listdir(bg_subdir_path):
        try:
            # 使用 OpenCV 打开图像
            image = cv2.imread(os.path.join(bg_subdir_path,file_name))

            # 检查图像大小是否符合要求
            if image.shape[1] != patch_size[1] or image.shape[0] != patch_size[0]:
                # 计算需要填充的高度和宽度
                h_diff = patch_size[0] - image.shape[0]
                w_diff = patch_size[1] - image.shape[1]

                # 只填充右边和下边，其余部分保持不变
                top = 0
                bottom = h_diff
                left = 0
                right = w_diff

                # 使用 cv2.copyMakeBorder() 填充图像块，将右边和下边的剩余空间填充为白色（255）
                cur_region = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                value=(255, 255, 255))

                #这里可以将 cur_region 保存为新的图像文件，如果需要的话
                cv2.imwrite(os.path.join(target_subdir_path,file_name), cur_region)
                print("tianchong",file_name)
            else:
                shutil.copy(os.path.join(bg_subdir_path,file_name),os.path.join(target_subdir_path,file_name))
        except Exception as e:
                # 捕获异常
                print(f"处理文件 {file_name} 时出错: {e}")