import os
from PIL import Image

# 设置路径
input_dir = '/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/results/evaluate_100bubble_patch'      # 输入图目录
result_dir = '/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/results/bubble_newcycle_add41bg/test_150/evaluate_100bubble_patch'    # 结果图目录
output_dir = '/home/xingzehang/project_18t/xzh/project/FFPEPlus-main2023/results/bubble_newcycle_add41bg/test_150/evaluate_100bubble_patch_processsEdge'    # 输出目录
brightness_threshold = 3            # 亮度阈值（0~255）

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有图像文件
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    result_path = os.path.join(result_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # 检查是否是对应格式的图像
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    if not os.path.exists(result_path):
        print(f"警告：结果图中找不到对应文件 {filename}，跳过")
        continue

    # 打开图像
    input_img = Image.open(input_path).convert('RGB')
    result_img = Image.open(result_path).convert('RGB')

    # 确保图像尺寸一致
    if input_img.size != result_img.size:
        print(f"警告：尺寸不一致，跳过 {filename}")
        continue

    # 创建新图像
    new_img = Image.new('RGB', input_img.size)
    pixels_in = input_img.load()
    pixels_res = result_img.load()
    pixels_new = new_img.load()

    # 遍历每个像素
    for y in range(input_img.height):
        for x in range(input_img.width):
            r, g, b = pixels_in[x, y]
            brightness = (r + g + b) / 3
            if brightness < brightness_threshold:
                pixels_new[x, y] = (0, 0, 0)  # 变黑
            else:
                pixels_new[x, y] = pixels_res[x, y]

    # 保存处理后的图像
    new_img.save(output_path)
    print(f"已处理：{filename}")

print("全部处理完成。")