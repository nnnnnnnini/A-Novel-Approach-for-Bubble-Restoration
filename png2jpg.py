import os
import psutil
from PIL import Image
import time


def convert_png_to_jpg(png_path, jpg_path):
    try:
        print(png_path)
        img = Image.open(png_path)
        # Resize the image to a smaller dimension
        max_dimension = 65500
        if max(img.size) > max_dimension:
            img.thumbnail((max_dimension, max_dimension))
        jpg_path = jpg_path.replace(".png", ".jpg")  # Update the filename to .jpg extension
        img.save(jpg_path, "JPEG")
        print(f"Converted {png_path} to {jpg_path}")
    except Exception as e:
        print(f"Error converting {png_path}: {e}")
def print_memory_usage():
    process = psutil.Process()

    while True:
        memory_info = process.memory_info()
        print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        time.sleep(1)  # Print memory usage every second
import threading
memory_thread = threading.Thread(target=print_memory_usage)
memory_thread.daemon = True
memory_thread.start()

# Replace this with the path to your folder
folder = "/home/xingzehang/project_100t/tuisexiufu_results/ffpep_cyclegan/test_50/wsi_512/fakeB_fenlei/"
Image.MAX_IMAGE_PIXELS = None


for filename in os.listdir(folder):
    if filename.endswith(".png"):
        png_path = os.path.join(folder, filename)
        jpg_path = os.path.join(folder, filename.replace(".png", ".jpg"))

        if not os.path.exists(jpg_path):
            print('"'+png_path+" "+ jpg_path + '"',)
            convert_png_to_jpg(png_path, jpg_path)