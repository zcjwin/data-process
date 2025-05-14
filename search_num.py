import json
import os
import shutil

import yaml
from tqdm import tqdm


def key_label_json():
    with open('mydata.yaml', 'r', encoding='utf-8', errors='ignore') as f:
        data = yaml.load(f, Loader=yaml.Loader)

    # print(data['names'])
    dir_path = 'labels'
    data_dict = {}
    for key, value in data['names'].items():
        print(key, value)
        data_dict[key] = []
        for file in os.listdir(dir_path):
            if file.endswith(".txt"):
                with open("labels/" + file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for line in lines:
                        if int(line.split(' ')[0]) == int(key):
                            # print(file)
                            print(line)
                            # data_dict[key].append(os.path.join(dir_path, file))
                            data_dict[key].append(file)

    with open('res.json', 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


def key_dir_file():
    with open('mydata.yaml', 'r', encoding='utf-8', errors='ignore') as ff:
        yaml_data = yaml.load(ff, Loader=yaml.Loader)
    with open('res.json', 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)

    for key, value in data.items():
        if not os.path.exists(f'labels/{key}'):
            os.mkdir(f'labels/{key}')
        if not os.path.exists(f'images/{key}'):
            os.mkdir(f'images/{key}')
        for file in tqdm(value):
            if not os.path.exists(f'labels\\{file}'):
                continue
            if os.path.exists(f'labels\\{key}\\{file}'):
                continue
            shutil.move(f'labels\\{file}', f'labels\\{key}')
            img = file.replace('.txt', '.jpg')
            if os.path.exists(f'images\\{key}\\{img}'):
                continue
            shutil.move(f'images\\{img}', f'images\\{key}')
    # for key, value in data.items():
    #     key = int(key)
    #     if not os.path.exists(f"labels/{yaml_data['names'][key]}"):
    #         os.mkdir(f"labels/{yaml_data['names'][key]}")
    #     if not os.path.exists(f"images/{yaml_data['names'][key]}"):
    #         os.mkdir(f"images/{yaml_data['names'][key]}")
    #     for file in tqdm(value):
    #         if not os.path.exists(f'labels\\{file}'):
    #             continue
    #         if os.path.exists(f'labels\\{key}\\{file}'):
    #             continue
    #         shutil.move(f'labels\\{file}', f"labels/{yaml_data['names'][key]}")
    #         img = file.replace('.txt', '.jpg')
    #         if os.path.exists(f'images\\{key}\\{img}'):
    #             continue
    #         shutil.move(f'images\\{img}', f"images/{yaml_data['names'][key]}")


def file_list(dir_path):
    data = []
    for file in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, file)):
            data += file_list(os.path.join(dir_path, file))
        elif file.endswith(".txt"):
            data.append(os.path.join(dir_path, file))
        # else:
        #     file_num = len(os.listdir(dir_path))
        #     print(file_num)
    return data



def classify_num():
    with open('mydata.yaml', 'r', encoding='utf-8', errors='ignore') as f:
        data = yaml.load(f, Loader=yaml.Loader)

    label_path = r'E:\yolo\data-process\labels'
    data_dict = {}
    data_list = []
    for key, value in data['names'].items():
        print(key, value)
        data_dict[key] = []
        data_list.append(value)
    print(data_list)
    for file in tqdm(os.listdir(label_path)):
        # if file != "0010000054459.txt":
        #     continue
        if file.endswith(".txt"):
            with open(os.path.join(label_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                try:
                    index = int(lines[0].split(' ')[0])
                except Exception as e:
                    print(e)
                    continue
                data_dict[index].append(file)
    print(data_dict)
    with open('res.json', 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

    with open('res.json', 'r', encoding='utf-8', errors='ignore') as f2:
        data_dict = json.load(f2)
    AUG_IMAGE_PATH = 'aug/images'
    AUG_LABEL_PATH = 'aug/labels'
    if os.path.exists(AUG_IMAGE_PATH):
        shutil.rmtree(AUG_IMAGE_PATH)
    if os.path.exists(AUG_LABEL_PATH):
        shutil.rmtree(AUG_LABEL_PATH)
    os.makedirs(AUG_IMAGE_PATH, exist_ok=True)
    os.makedirs(AUG_LABEL_PATH, exist_ok=True)
    for k, v in data_dict.items():
        if 300 < len(v) < 400:
            print(f"index: {k}, count: {len(v)}")
            # if not os.path.exists(aug_dir):
            #     os.mkdir(aug_dir)
            for file in v:
                if not os.path.exists(f"{AUG_IMAGE_PATH}/{k}"):
                    os.mkdir(f"{AUG_IMAGE_PATH}/{k}")
                if not os.path.exists(f"{AUG_LABEL_PATH}/{k}"):
                    os.mkdir(f"{AUG_LABEL_PATH}/{k}")
                # 获取文件扩展名
                _, file_extension = os.path.splitext(file)
                # 确保扩展名以小写形式存储
                file_extension = file_extension.lower()
                possible_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp", ".JPG", ".JPEG", ".PGN"]
                for ext in possible_extensions:
                    # 构建完整文件名
                    file_name = _ + ext
                    # 检查文件是否存在于directory目录下
                    if os.path.exists(os.path.join('images', file_name)):
                        # target_path = os.path.join(aug_dir, 'images', f"{k}{file_name}")
                        # shutil.copy(f'images/{file_name}', f"{aug_dir}/images/{k}")
                        shutil.copy(f'images/{file_name}', f'{AUG_IMAGE_PATH}/{k}')

                # 构建目标文件路径，包含正确的扩展名
                # target_path = os.path.join(aug_dir, 'images', f"{k}{_}")

                # 复制文件
                # shutil.copy(f'images\\{file[13:]}', target_path)
                # shutil.copy(f'labels/{file}', f"{aug_dir}/labels/{k}")
                shutil.copy(f'labels/{file}', f'{AUG_LABEL_PATH}/{k}')


if __name__ == '__main__':
    # key_label_json()
    # key_dir_file()
    # file = file_list(os.path.join(r"D:\MyWork\yolo\datautils", "labels"))
    # for i in tqdm(file):
    #     print(i[-17:])
    #     aug_file = f'dataset/object_detection/labels_aug/{i[-17:]}'
    #     if os.path.exists(aug_file):
    #         continue
    #     shutil.move(i, 'dataset/object_detection/labels_aug')
    #     # shutil.move(i.replace('.jpg', '.txt'), 'dataset/object_detection/labels_aug')

    classify_num()
