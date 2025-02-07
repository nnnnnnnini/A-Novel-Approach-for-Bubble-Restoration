import os
import re


def rename_images_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                file_name, file_extension = os.path.splitext(file)

                # 使用正则表达式从文件名中提取部分
                parts = file_name.split('_')

                # 如果文件名中包含至少三个部分且最后两部分可以转换为数字
                if len(parts) >= 3 :
                    new_name = parts[0] + "_" + str(int(parts[-2])*256) + "_" + str(int(parts[-1])*256) + file_extension
                    new_file_path = os.path.join(root, new_name)

                    # 重命名文件
                    os.rename(file_path, new_file_path)
                    print(f"Renamed: {file_path} to {new_file_path}")


# 指定要处理的目录路径
directory_path = "/home/xingzehang/project_100t/tuisexiufu_results/ffpep_cut/test_40/fakeB_fenlei"

# 调用函数对目录下的图片文件进行重命名
rename_images_in_directory(directory_path)