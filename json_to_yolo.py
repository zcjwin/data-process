import json
import os
import shutil
import time
from pathlib import Path

from PIL import Image
import concurrent.futures
from tqdm import tqdm


# ----------------------------
# 配置参数
# ----------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp", ".JPG", ".JPEG", ".PGN"}
DATASET_ROOT = r"D:\MyWork\datasets"
OUTPUT_DIR = Path(__file__).parent
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"
CONFIG_YAML_PATH = OUTPUT_DIR / "mydata.yaml"
YOLO11_CONFIG_PATH = OUTPUT_DIR / "yolo11-test.yaml"


def get_timestamp():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def file_list(top_path):
    json_list = []
    image_list = []
    for dirpath, dirnames, filenames in os.walk(top_path):
        # print(f"当前目录: {dirpath}")
        # print("子目录:")
        # for dirname in dirnames:
        #     print(f"  {dirname}")
        # print("文件:")
        for filename in filenames:
            # print(f"  {filename}")
            if filename.endswith(".json"):
                json_list.append(os.path.join(dirpath, filename))
            elif is_image_file(filename):
                image_list.append(os.path.join(dirpath, filename))
        # print("-" * 20)
    return json_list, image_list


# ----------------------------
# 标注转换逻辑
# ----------------------------
def convert_polygon_to_bbox(img_size, box):
    x1_center = box[0] + (box[2] - box[0]) / 2.0
    y1_center = box[1] + (box[3] - box[1]) / 2.0

    w_1 = box[2] - box[0]
    h_1 = box[3] - box[1]

    x1_normal = x1_center / img_size[0]
    y1_normal = y1_center / img_size[1]

    w_1_normal = w_1 / img_size[0]
    y_1_normal = h_1 / img_size[1]

    return (x1_normal, y1_normal, w_1_normal, y_1_normal)


def convert_json_to_yolo(json_path, label_list, index):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    image_name = json_data["imagePath"]
    image_w = json_data["imageWidth"]
    image_h = json_data["imageHeight"]
    # 构建新文件名（13位补零）
    new_filename = f"{index:013d}"
    ext = os.path.splitext(image_name)[1]  # 保留原始扩展名
    new_image_name = f"{new_filename}{ext}"
    new_label_name = f"{new_filename}.txt"
    image_path = IMAGES_DIR / new_image_name
    label_path = LABELS_DIR / new_label_name

    # 构造源图片路径：基于 JSON 文件所在目录
    json_dir = Path(json_path).parent
    src_image_path = json_dir / image_name
    # 复制图片
    if src_image_path.exists():
        shutil.copy(src_image_path, image_path)

    with open(label_path, 'w', encoding='utf-8') as out_f:
        for shape in json_data["shapes"]:
            label = shape["label"]
            if label not in label_list:
                continue
                # label_list.append(label)
            index = label_list.index(label)
            # 仅处理矩形标注
            if shape["shape_type"] != "rectangle":
                continue
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[2]
            bbox = convert_polygon_to_bbox((image_w, image_h), (x1, y1, x2, y2))
            out_f.write(f"{index} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


def copy_negative_sample(img_path, index):
    ext = os.path.splitext(img_path)[1]
    new_name = f"{index:013d}{ext}"
    new_path = IMAGES_DIR / new_name
    if not new_path.exists():
        shutil.copy(img_path, new_path)


# ----------------------------
# 数据集处理逻辑
# ----------------------------
def collect_labels(json_list):
    label_set = set()
    for json_path in tqdm(json_list, desc="收集标签"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data['shapes']:
            label_set.add(shape['label'])
    return sorted(label_set)


def generate_classes_yaml(label_list):
    with open(OUTPUT_DIR / 'classes.txt', 'w', encoding='utf-8') as f:
        f.write(f"nc: {len(label_list)}\n")
        f.write("names:\n")
        for idx, name in enumerate(label_list):
            f.write(f"  {idx}: {name}\n")


def generate_yaml_config(label_list):
    with open(CONFIG_YAML_PATH, "w", encoding="utf-8") as f:
        f.write(
            "# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]\n"
            "path: /root/autodl-tmp/datautils/datasets/ # dataset root dir\n"
            "train: train/images # train images (relative to 'path') 118287 images\n"
            "val: val/images # val images (relative to 'path') 5000 images\n"
            "test: test/images # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794\n\n"
        )
        f.write("names:\n")
        for idx, name in enumerate(label_list):
            f.write(f"  {idx}: {name}\n")

    with open(YOLO11_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(
            "type: yolo11\n"
            "name: yolo11s-r20240930\n"
            "display_name: YOLO11s Ultralytics\n"
            "model_path: .\\best.onnx\n"
            "nms_threshold: 0.45\n"
            "confidence_threshold: 0.25\n"
        )
        f.write("classes:\n")
        for name in label_list:
            f.write(f"  - {name}\n")


def main():
    # 初始化目录
    for d in [IMAGES_DIR, LABELS_DIR]:
        if d.exists():
            shutil.rmtree(d)
            print(f"删除旧目录: {d}")
        d.mkdir()

    # 收集文件
    json_list, image_list = file_list(DATASET_ROOT)
    total_files = len(json_list)
    print(f"共发现 {total_files} 个 JSON 文件 {len(image_list)} 张图片")

    # 收集正负样本
    json_basenames = {os.path.splitext(os.path.basename(j))[0] for j in json_list}
    negative_samples = [
        img for img in image_list
        if os.path.splitext(os.path.basename(img))[0] not in json_basenames
    ]

    # 收集标签并生成配置
    label_list = collect_labels(json_list)
    generate_classes_yaml(label_list)
    generate_yaml_config(label_list)

    # # 顺序处理正样本
    # print("处理正样本...")
    # for idx, json_path in enumerate(tqdm(json_list, desc="处理正样本")):
    #     convert_json_to_yolo(
    #         json_path=json_path,
    #         label_list=label_list,
    #         index=idx + 1,
    #     )
    #
    # # 顺序处理负样本
    # print("处理负样本...")
    # for idx, img_path in enumerate(tqdm(negative_samples, desc="处理负样本")):
    #     copy_negative_sample(
    #         img_path=img_path,
    #         index=idx + total_files + 1,
    #     )

    # 多线程处理正样本
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(convert_json_to_yolo, json_path, label_list, idx + 1)
            for idx, json_path in enumerate(json_list)
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="处理正样本"):
            future.result()

    # 多线程处理负样本
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(copy_negative_sample, img_path, idx + total_files + 1)
            for idx, img_path in enumerate(negative_samples)
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(negative_samples), desc="处理负样本"):
            future.result()


if __name__ == '__main__':
    main()
