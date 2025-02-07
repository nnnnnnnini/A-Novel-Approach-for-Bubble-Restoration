import os
from PIL import Image
import glob
import os
from PIL import Image
import openslide
from openslide.deepzoom import DeepZoomGenerator
import glob
import numpy as np
def doit(input_folder):
    print(f"processing {input_folder}")
    foldername = input_folder.split("/")[-1]

    base_ouput_folder = "/home/xingzehang/project_100t/tuisexiufu_results/ffpep_cyclegan/test_50/wsi_512/"
    os.makedirs(base_ouput_folder,exist_ok=True)

    output_folder = os.path.join(base_ouput_folder, input_folder.split("/")[-2])
    os.makedirs(output_folder, exist_ok=True)



    files_result = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    max_height = -1
    max_width = -1

    for file_name in files_result:
        parts = file_name.split(".")[0].split("_")
        if len(parts) >= 3:
            height = int(parts[-1])
            width = int(parts[-2])
            max_height = max(max_height, height)
            max_width = max(max_width, width)

    print("最大的height:", max_height)
    print("最大的width:", max_width)

    patch_size = 512
    overlap_size = 256
    new_image_result = Image.new("RGB",(max_width+patch_size,max_height+patch_size))
    for file_name in files_result:
        file_path = os.path.join(input_folder,file_name)
        image = Image.open(file_path)
        local_x = int(file_name.split(".")[-2].split("_")[-2])
        local_y = int(file_name.split(".")[-2].split("_")[-1])
        if local_x % patch_size == 0 and local_y % patch_size == 0:  #######正常
            new_image_result.paste(image,(local_x,local_y))
        elif local_x % patch_size == 0 and local_y % patch_size != 0:
            # 裁剪掉顶部的一半重叠区域
            crop_area = image.crop((0, overlap_size // 2, patch_size, patch_size))
            new_image_result.paste(crop_area, (local_x, local_y + overlap_size // 2))

        elif local_x % patch_size != 0 and local_y % patch_size == 0:
            # 裁剪掉左侧的一半重叠区域
            crop_area = image.crop((overlap_size // 2, 0, patch_size, patch_size))
            new_image_result.paste(crop_area, (local_x + overlap_size // 2, local_y))
        else:
            # 同时裁剪掉顶部和左侧的重叠区域
            crop_area = image.crop((overlap_size // 2, overlap_size // 2, patch_size, patch_size))
            new_image_result.paste(crop_area, (local_x + overlap_size // 2, local_y + overlap_size // 2))

    new_image_result.save(os.path.join(output_folder,f"{foldername}.png"))


import time
import psutil
def print_memory_usage():
    process = psutil.Process()

    while True:
        memory_info = process.memory_info()
        print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        time.sleep(1)  # Print memory usage every second
if __name__ == "__main__":
    import threading

    memory_thread = threading.Thread(target=print_memory_usage)
    memory_thread.daemon = True
    memory_thread.start()
    path = '/home/xingzehang/project_100t/tuisexiufu_results/ffpep_cyclegan/test_50'
    for subdir in os.listdir(path):
        if subdir != "fakeB_fenlei":
            continue
        subdir_path = os.path.join(path,subdir)
        for subsubdir in os.listdir(subdir_path):
            subsubdir_path = os.path.join(subdir_path,subsubdir)
            doit(subsubdir_path)

