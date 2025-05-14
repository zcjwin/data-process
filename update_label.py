import json
import os
import shutil

import pandas as pd
import numpy as np
import cv2
import yaml
from tqdm import tqdm


def data_process():
    data_path = r"D:\MyWork\datasets\一类\蔬菜蓟马\西花蓟马"
    # data_path = r"C:\Users\auto\Desktop\test\labels"
    json_file_list = []
    for file in os.listdir(data_path):
        if file.endswith(".json"):
            # print(file)
            file_path = os.path.join(data_path, file)
            json_file_list.append(file_path)
    print(len(json_file_list))
    for json_file in tqdm(json_file_list):
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # 修改 imagePath 的值
        new_image_path = os.path.basename(json_data['imagePath'])  # 这里替换为你想要的新路径
        print(f"原路径: {json_data['imagePath']}, 新路径: {new_image_path}")
        json_data['imagePath'] = new_image_path
        #     for shape in json_data["shapes"]:
        #         shape["label"] = "玉米螟"
        with open(json_file, "w", encoding="utf-8") as ff:
            json.dump(json_data, ff, ensure_ascii=False, indent=4)


def move_data():
    with open("label.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    for data in tqdm(json_data):
        _path = r"E:\yolo\datasets"
        label = data["label"]
        from_path = os.path.join(_path, data["image_path"])
        split_name = os.path.splitext(os.path.basename(data["image_path"]))
        json_path = from_path.replace(split_name[1], ".json")
        try:
            os.remove(data["image_path"])
        except Exception as e:
            print(e)
        try:
            os.remove(json_path)
        except Exception as e:
            print(e)


        # if not os.path.exists(os.path.join(_path, label)):
        #     os.mkdir(os.path.join(_path, label))
        # from_path = os.path.join(_path, data["image_path"])
        # split_name = os.path.splitext(os.path.basename(data["image_path"]))
        # json_path = from_path.replace(split_name[1], ".json")
        # to_path = os.path.join(_path, label)
        # print(f"复制 {from_path} 到 {to_path}")

from pypinyin import lazy_pinyin, Style, pinyin_dict
def chinese_to_abbreviation(name):
    pinyin_list = []
    for char in name:
        if '\u4e00' <= char <= '\u9fff':  # 判断是否为中文字符
            pinyin_list.append(lazy_pinyin(char, style=Style.FIRST_LETTER)[0])
        else:
            pinyin_list.append(char)  # 保留非中文字符
    return ''.join(pinyin_list).upper()


def not_valid_data_process():
    with open("mydata.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.load(f, Loader=yaml.Loader)
    names = yaml_data["names"]
    # name_list = []
    name_mapp = {}
    for index, name in names.items():
        pinyin_name = f"{index}{chinese_to_abbreviation(name)}"
        name_mapp[pinyin_name] = name
        # name_list.append(chinese_to_abbreviation(name))
    name_list = list(name_mapp.keys())
    print(name_list)
    # with open("not_valid_list.json", "r", encoding="utf-8") as f:
    #     datas = json.load(f)
    # for data in tqdm(datas):
    #     try:
    #         os.remove(data)
    #         print(f"删除 {data}")
    #     except Exception as e:
    #         print(e)
    #     split_name = os.path.splitext(os.path.basename(data))
    #     json_path = data.replace(split_name[1], ".json")
    #     try:
    #         os.remove(json_path)
    #         print(f"删除 {json_path}")
    #     except Exception as ee:
    #         print(ee)

        # _path = r"E:\yolo\datasets"
        # folder_path = os.path.dirname(data)
        # category = os.path.basename(folder_path)
        # target_path = os.path.join(_path, category)
        # os.makedirs(target_path, exist_ok=True)
        # shutil.copy(data, target_path)
        # split_name = os.path.splitext(os.path.basename(data))
        # json_path = data.replace(split_name[1], ".json")
        # shutil.copy(json_path, target_path)
        # print(f"复制 {data} 到 {target_path}")


if __name__ == '__main__':
    # data_process()
    # move_data()
    not_valid_data_process()
