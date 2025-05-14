# 批量json转yolo
import datetime
import hashlib
import json
import os, shutil
import uuid

from tqdm import tqdm
import concurrent.futures

import object_detection_data_aug

label = []
PATH = os.path.abspath(os.path.dirname(__file__))
num = 0
# num = str(uuid.uuid4()).split('-')[-1:][0]

if os.path.exists(os.path.join(PATH, "labels")):
    print("删除labels文件夹")
    shutil.rmtree(os.path.join(PATH, "labels"))

os.mkdir(os.path.join(PATH, "labels"))

if os.path.exists(os.path.join(PATH, "images")):
    print("删除images文件夹")
    shutil.rmtree(os.path.join(PATH, "images"))
os.mkdir(os.path.join(PATH, "images"))


def convert(img_size, box):
    x1_center = box[0] + (box[2] - box[0]) / 2.0
    y1_center = box[1] + (box[3] - box[1]) / 2.0

    w_1 = box[2] - box[0]
    h_1 = box[3] - box[1]

    x1_normal = x1_center / img_size[0]
    y1_normal = y1_center / img_size[1]

    w_1_normal = w_1 / img_size[0]
    h_1_normal = h_1 / img_size[1]

    return (x1_normal, y1_normal, w_1_normal, h_1_normal)


original_filename = "0000000000000.txt"
filename_length = len(original_filename) - 4

shape_type_list = []
not_valid_list = []
def decode_json(json_path):
    global num, label
    txt_path = os.path.join(PATH, "labels", f"{str(num).zfill(filename_length)}.txt")
    label_text = []
    json_path = os.path.join(json_path)
    # # 读取json文件
    data = json.load(open(json_path, "r", encoding="utf-8"))
    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    # 图片位置
    image_path = os.path.join(os.path.dirname(json_path), data["imagePath"])
    MOVE = False
    for i in data["shapes"]:
        # MOVE = True
        if i["label"] not in label:
            label.append(i["label"])
        label_index = label.index(i["label"])
        if i["shape_type"] == "rectangle":
            MOVE = True
            x1 = float(i["points"][0][0])
            y1 = float(i["points"][0][1])
            x2 = float(i["points"][2][0])
            y2 = float(i["points"][2][1])
            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            label_text.append(
                "{}".format(label_index) + " " + " ".join([str(i) for i in bbox]) + "\n"
            )
        else:
            type_dict = {
                "image_path": image_path,
                "shape_type": i["shape_type"],
                "label": i["label"],
            }
            print("不是矩形框：", type_dict)
            shape_type_list.append(type_dict)
    if MOVE:
        if os.path.exists(image_path):
            with open(txt_path, "w") as f:
                f.writelines(label_text)
            shutil.copyfile(
                image_path,
                os.path.join(
                    # PATH, "images", "{}{}".format(num, os.path.splitext(image_path)[-1])
                    PATH, "images", "{}{}".format(str(num).zfill(filename_length), os.path.splitext(image_path)[-1])
                    # PATH, "images", "{}.jpg".format(str(num).zfill(filename_length))
                ),
            )
            num += 1
            # num = str(uuid.uuid4()).split('-')[-1:][0]
        else:
            print("{}  文件不存在".format(image_path))
    else:
        print("{}  MOVE文件不存在".format(image_path))
        not_valid_list.append(image_path)


def file_list(dir_path):
    data = []
    for file in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, file)):
            data += file_list(os.path.join(dir_path, file))
        elif file.endswith(".json"):
            data.append(os.path.join(dir_path, file))
        # else:
        #     file_num = len(os.listdir(dir_path))
        #     print(file_num)
    return data


if __name__ == "__main__":

    # file = file_list(os.path.join(PATH, "AI"))
    file = file_list(os.path.join(r"D:\MyWork", "datasets"))

    for i in tqdm(file, unit=" 张"):
        # 去除json文件中imagePath中的空格
        with open(i, "r", encoding="utf-8") as ff:
            data = json.load(ff)
        # 修改 imagePath 的值
        # new_image_path = data['imagePath'].strip()  # 这里替换为你想要的新路径
        # data['imagePath'] = new_image_path
        # ff.seek(0)  # 移动文件指针到开头
        # json.dump(data, ff, ensure_ascii=False, indent=4)
        # ff.truncate()  # 清除多余内容
        #
        # # # 将修改后的内容写回文件
        # # with open(i, "w", encoding="utf-8") as ff:
        # #     json.dump(data, ff, ensure_ascii=False, indent=4)
        decode_json(i)
    print("不是矩形框数量：", len(shape_type_list))
    with open("label.json", "w", encoding="utf-8") as fs:
        json.dump(shape_type_list, fs, ensure_ascii=False, indent=4)
    print("未移动文件数量：",  len(not_valid_list))
    with open("not_valid_list.json", "w", encoding="utf-8") as fn:
        json.dump(not_valid_list, fn, ensure_ascii=False, indent=4)
    with open(os.path.join(PATH, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("nc: {}\n".format(len(label)))
        f.write("names:\n")
        # for i in label:
        #     f.write("  - {}\n".format(i))
        for index, name in enumerate(label):
            f.write("  {}: {}\n".format(index, name))
    # with open(os.path.join(PATH, "yolov9_mydata.yaml"), "w", encoding="utf-8") as f:
    #     f.write("type: yolov9\nname: yolov9-s\ndisplay_name: yolov9\nmodel_path: .\\best.onnx\nconfidence_threshold: 0.25\n")
    #     f.write("names:\n")
    #     for i in label:
    #         f.write("  - {}\n".format(i))
    # with open(os.path.join(PATH, "mydata.yaml"), "w", encoding="utf-8") as f2:
    #     f2.write(
    #         "# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]\n"
    #         # "path: /hy-tmp/datasets/ # dataset root dir\n"
    #         "#path: D:/MyWork/yolo/datautils/datasets/ # dataset root dir\n"
    #         "train: /hy-tmp/datasets/images/train # train images (relative to 'path') 118287 images\n"
    #         "val: /hy-tmp/datasets/images/val # val images (relative to 'path') 5000 images\n"
    #         "test: /hy-tmp/datasets/images/test # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794\n\n"
    #     )
    #     f2.write("names:\n")
    #     for index, name in enumerate(label):
    #         f2.write("  {}: {}\n".format(index, name))
    # yolo11使用
    with open(os.path.join(PATH, "yolo11-test.yaml"), "w", encoding="utf-8") as f:
        f.write("type: yolo11\nname: yolo11s-r20240930\ndisplay_name: YOLO11s Ultralytics\nmodel_path: .\\best.onnx\nnms_threshold: 0.45\nconfidence_threshold: 0.25\n")
        f.write("classes:\n")
        for i in label:
            f.write("  - {}\n".format(i))
    with open(os.path.join(PATH, "mydata.yaml"), "w", encoding="utf-8") as f2:
        f2.write(
            "# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]\n"
            "path: /root/autodl-tmp/datautils/datasets/ # dataset root dir\n"
            "#path: D:/MyWork/yolo/datautils/datasets/ # dataset root dir\n"
            "train: train/images # train images (relative to 'path') 118287 images\n"
            "val: val/images # val images (relative to 'path') 5000 images\n"
            "test: test/images # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794\n\n"
        )
        f2.write("names:\n")
        for index, name in enumerate(label):
            f2.write("  {}: {}\n".format(index, name))
