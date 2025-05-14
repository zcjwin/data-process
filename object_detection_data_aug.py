import warnings

warnings.filterwarnings('ignore')
import os, shutil, cv2, tqdm
import numpy as np
import albumentations as A
from PIL import Image
from multiprocessing import Pool
from typing import Callable, Dict, List, Union

# https://github.com/albumentations-team/albumentations
# https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#geometric-transforms-augmentationsgeometrictransforms:~:text=Contributing%20to%20Albumentations-,Geometric%20transforms%20(augmentations.geometric.transforms),-%C2%B6

# IMAGE_PATH = './aug/images'
# LABEL_PATH = './aug/labels'
# AUG_IMAGE_PATH = 'datasets/object_detection/images_aug'
# AUG_LABEL_PATH = 'datasets/object_detection/labels_aug'
AUG_IMAGE_PATH = 'images'
AUG_LABEL_PATH = 'labels'
SHOW_SAVE_PATH = 'results'
# 类别名称列表和mydata.yaml中的names对应，这里使用了索引加中文缩写，因增强后画框验证使用的是cv2，cv不支持中文，故使用缩写代替中文
CLASSES = ['0XJ', '1PG', '2L']
ENHANCEMENT_LOOP = 10
ENHANCEMENT_STRATEGY = A.Compose([
    A.Compose([
        A.Affine(
            scale=[0.5, 1.5],
            translate_percent=[0.0, 0.3],
            rotate=[-30, 30],
            shear=[0, 15],
            keep_ratio=True,
            p=0.1
        ),  # 对输入进行仿射变换，包括缩放、平移、旋转和错切
        A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.1),  # 在不丢失bbox的情况下裁剪输入的随机部分.
        A.D4(p=0.1),
        # 将八种可能的D4二面角群变换之一应用于方形输入，保持方形。这些变换对应于正方形的对称性，包括旋转和反射.
        A.ElasticTransform(p=0.1),  # Elastic deformation of images as described in [Simard2003]_ (with modifications).
        A.Flip(p=0.1),  # 水平、垂直或同时水平和垂直翻转输入
        A.GridDistortion(p=0.1),
        # 将网格失真增强应用于图像、遮罩和边界框。该技术涉及将图像划分为单元格网格，并随机移动网格的交点，从而导致局部失真
        A.Perspective(p=0.1),  # 对输入执行随机的四点透视变换
    ], p=1.0),

    A.Compose([
        # A.GaussNoise(p=0.1),  # 对图片给进行高斯噪声
        A.ISONoise(p=0.1),  # 应用摄像头传感器噪声
        A.ImageCompression(quality_lower=50, quality_upper=100, p=0.1),  # 压缩图像来降低图像质量
        A.RandomBrightnessContrast(p=0.1),  # 随机更改输入图像的亮度和对比度.
        # A.RandomFog(p=0.1),  # 模拟图像的雾
        A.RandomRain(p=0.1),  # 为图像添加雨水效果.
        # A.RandomSnow(p=0.1),  # Bleach out some pixel values imitating snow.
        A.RandomShadow(p=0.1),  # 模拟图像的阴影
        # A.RandomSunFlare(p=0.1),  # 为图像模拟太阳耀斑
        # A.ToGray(p=0.1),  # Convert the input RGB images to grayscale
    ], p=1.0)

    # A.OneOf([
    #     A.GaussNoise(p=1.0), # Apply Gaussian noise to the input images.
    #     A.ISONoise(p=1.0), # Apply camera sensor noise.
    #     A.ImageCompression(quality_lower=50, quality_upper=100, p=1.0), # Decreases images quality by Jpeg, WebP compression of an images.
    #     A.RandomBrightnessContrast(p=1.0), # Randomly change brightness and contrast of the input images.
    #     A.RandomFog(p=1.0), # Simulates fog for the images.
    #     A.RandomRain(p=1.0), # Adds rain effects to an images.
    #     A.RandomSnow(p=1.0), # Bleach out some pixel values imitating snow.
    #     A.RandomShadow(p=1.0), # Simulates shadows for the images
    #     A.RandomSunFlare(p=1.0), # Simulates Sun Flare for the images
    #     A.ToGray(p=1.0), # Convert the input RGB images to grayscale
    # ], p=1.0),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_labels']))


def parallelise(function: Callable, data: List, chunksize=100, verbose=True, num_workers=os.cpu_count()) -> List:
    num_workers = 1 if num_workers < 1 else num_workers  # Pool needs to have at least 1 worker.
    pool = Pool(processes=num_workers)
    results = list(
        tqdm.tqdm(pool.imap(function, data, chunksize), total=len(data), disable=not verbose)
    )
    pool.close()
    pool.join()
    return results


def draw_detections(box, name, img):
    height, width, _ = img.shape
    xmin, ymin, xmax, ymax = list(map(int, list(box)))

    # 根据图像大小调整矩形框的线宽和文本的大小
    line_thickness = max(1, int(min(height, width) / 200))
    font_scale = min(height, width) / 500
    font_thickness = max(1, int(min(height, width) / 200))
    # 根据图像大小调整文本的纵向位置
    text_offset_y = int(min(height, width) / 50)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), line_thickness)
    cv2.putText(img, str(name), (xmin, ymin - text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                font_thickness, lineType=cv2.LINE_AA)
    return img


def show_labels(images_base_path, labels_base_path):
    if os.path.exists(SHOW_SAVE_PATH):
        shutil.rmtree(SHOW_SAVE_PATH)
    os.makedirs(SHOW_SAVE_PATH, exist_ok=True)

    for images_name in tqdm.tqdm(os.listdir(images_base_path)):
        file_heads, _ = os.path.splitext(images_name)
        # images_path = f'{images_base_path}/{images_name}'
        images_path = os.path.join(images_base_path, images_name)
        # labels_path = f'{labels_base_path}/{file_heads}.txt'
        labels_path = os.path.join(labels_base_path, f'{file_heads}.txt')
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                labels = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float64), f.readlines())),
                                  dtype=np.float64)
            images = cv2.imread(images_path)
            height, width, _ = images.shape
            for cls, x_center, y_center, w, h in labels:
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                draw_detections([x_center - w // 2, y_center - h // 2, x_center + w // 2, y_center + h // 2],
                                CLASSES[int(cls)], images)
            # cv2.imwrite(f'{SHOW_SAVE_PATH}/{images_name}', images)
            cv2.imwrite(os.path.join(SHOW_SAVE_PATH, images_name), images)
            print(f'{SHOW_SAVE_PATH}/{images_name} save success...')
        else:
            print(f'{labels_path} label file not found...')


def data_aug_single(images_name):
    file_heads, postfix = os.path.splitext(images_name)
    # images_path = f'{IMAGE_PATH}/{images_name}'
    images_path = os.path.join(IMAGE_PATH, images_name)
    # labels_path = f'{LABEL_PATH}/{file_heads}.txt'
    labels_path = os.path.join(LABEL_PATH, f'{file_heads}.txt')
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float64), f.readlines())),
                              dtype=np.float64)
        images = Image.open(images_path)
        for i in range(ENHANCEMENT_LOOP):
            # new_images_name = f'{AUG_IMAGE_PATH}/{file_heads}_{i:0>3}{postfix}'
            # new_images_name = os.path.join(AUG_IMAGE_PATH, f'{file_heads}_{i:0>3}{postfix}')
            new_images_name = os.path.join(AUG_IMAGE_PATH, f'{i+1:0>3}{file_heads[3:]}{postfix}')
            # new_labels_name = f'{AUG_LABEL_PATH}/{file_heads}_{i:0>3}.txt'
            # new_labels_name = os.path.join(AUG_LABEL_PATH, f'{file_heads}_{i:0>3}.txt')
            new_labels_name = os.path.join(AUG_LABEL_PATH, f'{i+1:0>3}{file_heads[3:]}.txt')
            try:
                transformed = ENHANCEMENT_STRATEGY(image=np.array(images),
                                                   bboxes=np.minimum(np.maximum(labels[:, 1:], 0), 1),
                                                   class_labels=labels[:, 0])
            except:
                continue
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']

            cv2.imwrite(new_images_name, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
            with open(new_labels_name, 'w+') as f:
                for bbox, cls in zip(transformed_bboxes, transformed_class_labels):
                    f.write(f'{int(cls)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')
            print(f'{new_images_name} and {new_labels_name} save success...')
    else:
        print(f'{labels_path} label file not found...')


def data_aug(IMAGE_PATH):
    # if os.path.exists(AUG_IMAGE_PATH):
    #     shutil.rmtree(AUG_IMAGE_PATH)
    # if os.path.exists(AUG_LABEL_PATH):
    #     shutil.rmtree(AUG_LABEL_PATH)
    #
    # os.makedirs(AUG_IMAGE_PATH, exist_ok=True)
    # os.makedirs(AUG_LABEL_PATH, exist_ok=True)

    for images_name in tqdm.tqdm(os.listdir(IMAGE_PATH)):
        data_aug_single(images_name)


if __name__ == '__main__':
    """
    index: 35, count: 267
    index: 36, count: 295
    index: 38, count: 282
    index: 54, count: 268
    index: 56, count: 206
    index: 71, count: 294
    index: 73, count: 223
    index: 74, count: 271
    index: 84, count: 292
    index: 87, count: 250
    index: 90, count: 296
    index: 107, count: 202
    """
    image_path_index_list = [35, 36, 38, 54, 56, 73, 74, 84, 87, 90]
    for idx in image_path_index_list:
        IMAGE_PATH = f'./aug/images/{idx}'
        LABEL_PATH = f'./aug/labels/{idx}'
        data_aug(IMAGE_PATH)

    # show_labels(IMAGE_PATH, LABEL_PATH)
    # show_labels(AUG_IMAGE_PATH, AUG_LABEL_PATH)

