import os
import cv2
import json

def crop_and_save_from_json_dir(json_dir, image_dir, output_dir):
    """
    批量处理 JSON 文件，根据每个文件的标注，从原图中剪切目标区域并保存。

    Args:
        json_dir (str): 存放 JSON 文件的目录。
        image_dir (str): 原始图片存放目录。
        output_dir (str): 剪切后的图片保存目录。
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历 JSON 文件目录
    for json_file in os.listdir(json_dir):
        if not json_file.endswith('.json'):
            continue  # 跳过非 JSON 文件

        json_path = os.path.join(json_dir, json_file)

        # 处理每个 JSON 文件
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 获取图片路径和基本名称
        image_path = os.path.join(image_dir, data['imagePath'])
        image_name = os.path.splitext(data['imagePath'])[0]  # 去掉文件后缀

        # 加载原图
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image not found for JSON {json_path}")
            continue

        # 遍历 shapes 获取坐标和 group_id
        for i, shape in enumerate(data['shapes']):
            # 获取坐标
            points = shape['points']
            x_min, y_min = map(int, points[0])
            x_max, y_max = map(int, points[1])

            # 裁剪目标区域
            cropped_image = image[y_min:y_max, x_min:x_max]

            # 获取 group_id，如果没有则默认使用 0
            group_id = shape.get('label', 'unknown')

            # 构造保存路径
            save_name = f"{group_id}_{image_name}_{i}.jpg"
            save_path = os.path.join(output_dir, save_name)

            # 保存裁剪后的图片
            try:
                cv2.imwrite(save_path, cropped_image)
                print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Error: {e} for ", image_path)

# 使用示例
json_dir = "/home/jamie/Works/Waterloo/ECE730/pics_campus_labelme"    # 存放 JSON 文件的目录
image_dir = "/home/jamie/Works/Waterloo/ECE730/pics_campus_labelme"       # 存放原图的目录
output_dir = "/home/jamie/Works/Waterloo/ECE730/cut"      # 存放裁剪结果的目录

crop_and_save_from_json_dir(json_dir, image_dir, output_dir)
