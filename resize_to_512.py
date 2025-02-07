import os
from PIL import Image

def resize_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                width, height = image.size
                if width != 512 or height != 512:
                    print(f"Resizing {file}")
                    resized_image = image.resize((512, 512))
                    resized_image.save(image_path)

# 使用示例
resize_images('/home/xingzehang/project_16t/hxf/bubble_repair/qietu/testA/bubble_2023-12-11_12_30_29')