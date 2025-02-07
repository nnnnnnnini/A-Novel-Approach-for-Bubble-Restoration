import os
from PIL import Image

import glob

def doit(foldername):
    print(f"processing {foldername}")
    input_folder = f"/home/xzh/project/FFPEPlus-main/results/test_23/images/fake_Bfenlei/{foldername}/"
    ouput_folder = "/home/xzh/project/FFPEPlus-main/results/test_23/images/fake_Bfenlei_combination/"
    os.makedirs(ouput_folder,exist_ok=True)

    files_result = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    # print(files)
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
    patch_size = 128
    new_image_result = Image.new("RGB",(patch_size*((max_width+1)),patch_size*((max_height+1))))
    for file_name in files_result:
        file_path = os.path.join(input_folder,file_name)
        image = Image.open(file_path)
        #resized_image = image
        print(file_name)
        resized_image = image.resize((patch_size,patch_size), Image.BILINEAR)
        #_,local_x, local_y,__ = map(int, file_name.split(".")[0].split("_"))
        #temp = file_name.split(".")[-2].split("_")
        local_x = int(file_name.split(".")[-2].split("_")[-2])
        local_y = int(file_name.split(".")[-2].split("_")[-1])

        #local_x, local_y= map(int, file_name.split(".")[0].split("_"))
        local_x=local_x * patch_size
        local_y=local_y*patch_size

        new_image_result.paste(resized_image,(local_x,local_y))
        image.close()

    new_image_result.save(os.path.join(ouput_folder,f"{foldername}_fake_B.png"))

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
    path = '/home/xzh/project/FFPEPlus-main/results/test_23/images/fake_Bfenlei/'
    all_file = glob.glob(path+"*")
    print(all_file)
    for _ in all_file:
        foldername = _.split("/")[-1]
        doit(foldername)

