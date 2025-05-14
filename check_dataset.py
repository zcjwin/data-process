import logging

from PIL import Image
import os

from tqdm import tqdm

logger = logging.getLogger(__name__)


def check_image_integrity(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # 验证图像文件的完整性
            return True
    except Exception as e:
        logger.warning(f"图像文件损坏: {image_path}, 错误: {e}")
        return False


def check_dataset_integrity(dataset_path):
    labels_path = os.path.join(dataset_path, "labels")
    for label_file in tqdm(os.listdir(labels_path)):
        # if label_file != "0000000020131_003.txt":
        #     continue
        label_path = os.path.join(labels_path, label_file)
        file_size = os.path.getsize(label_path)
        if file_size < 1:
            print(f"{label_file} 文件大小: {file_size}")
            os.remove(label_path)
            img_path = label_path.replace("labels", "images").replace(".txt", ".jpg")
            if os.path.exists(img_path):
                logger.warning(f"{img_path} 文件中检测目标不存在")
                os.remove(img_path)
        # with open(label_path, 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip()
        #         if not line:
        #             continue
        #         label_info = line.split()
        #         if len(label_info) < 5:
        #             logger.warning(f"{label_file} 没有检测到类别信息")
    # for root, _, files in os.walk(dataset_path):
    #     for file in tqdm(files):
    #         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
    #             image_path = os.path.join(root, file)
    #             if not check_image_integrity(image_path):
    #                 # 处理损坏的图像文件，例如删除或修复
    #                 os.remove(image_path)
    #                 logger.warning(f"已删除损坏的图像文件: {image_path}")
    #             else:
    #                 # print(f"图像文件{image_path}正常")
    #                 pass


def clean_dataset(image_dir, supported_formats):
    for root, _, files in os.walk(image_dir):
        for file in files:
            if not any(file.lower().endswith(format) for format in supported_formats):
                print(f"Unsupported image format: {os.path.join(root, file)}")
                # 可以选择转换图像格式或删除该文件
                try:
                    img = Image.open(os.path.join(root, file))
                    new_path = os.path.splitext(os.path.join(root, file))[0] + '.jpg'
                    img.save(new_path, 'JPEG')
                    os.remove(os.path.join(root, file))
                    print(f"Converted to JPEG and saved as: {new_path}")
                except Exception as e:
                    print(f"Failed to convert image: {e}")
                    os.remove(os.path.join(root, file))  # 如果无法转换，则删除文件


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset_path = "D:/MyWork/yolo/datautils"
    check_dataset_integrity(dataset_path)

    # supported_image_formats = ['heic', 'jpeg', 'bmp', 'mpo', 'dng', 'tiff', 'pfm', 'png', 'tif', 'webp', 'jpg']
    # clean_dataset(r'D:\MyWork\yolo\datautils\datasets\train\images', supported_image_formats)

# 检查数据集的完整性
# check_dataset_integrity("D:/MyWork/yolo/datautils/datasets/val/images")
