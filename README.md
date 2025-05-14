# JSON转YOLO标注转换工具

## 📌 项目简介
本工具用于将LabelImg等工具生成的JSON标注文件批量转换为YOLO目标检测框架支持的格式，支持自动生成数据集配置文件，并可处理正负样本分离。

---

## 🚀 主要功能
- ✅ JSON标注文件转YOLO格式(txt)
- 📁 自动区分正样本（含标注的图像）和负样本（无标注的图像）
- 🔢 13位数字前缀重命名确保文件唯一性
- 📦 自动生成YOLO训练所需`classes.txt`和`mydata.yaml`配置文件
- ⚙️ 支持多线程加速处理（可切换为单线程模式）

---

## 🧩 目录结构要求
### 输入数据格式

```bash
DATASET_ROOT/ 
├── train/ 
│ ├── img1.jpg 
│ ├── img1.json 
│ ├── img2.jpg # 无对应JSON视为负样本 
├── val/ 
│ ├── ...
```

### 输出数据格式
```bash
OUTPUT_DIR/ 
├── images/ 
│ ├── 0000000000001.jpg # 正样本 
│ ├── 0000000000003.jpg # 负样本
├── labels/ 
│ ├── 0000000000001.txt # YOLO标注文件 
├── classes.txt # 标签映射文件 
├── mydata.yaml # YOLO训练配置 
└── yolo11-test.yaml # YOLO11模型配置
```

---

## 🛠️ 安装依赖
```bash
pip install json pathlib2 PIL tqdm concurrent.futures
```
---

## ⚙️ 配置参数
在`json_to_yolo.py`中修改以下参数：

```python
# 数据路径配置
DATASET_ROOT = r"D:\MyWork\datasets" # 原始数据根目录 OUTPUT_DIR = Path(file).parent # 输出目录
# 支持的图像格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
# YOLO配置文件生成路径
YOLO11_CONFIG_PATH = OUTPUT_DIR / "yolo11-test.yaml"
```
---

## ▶️ 使用指南
### 1. 数据准备
- 确保JSON文件与对应图像在同一目录
- 支持嵌套目录结构扫描

### 2. 运行转换
```python
python json_to_yolo.py
```

### 3. 多线程控制
- 当前默认启用多线程（推荐）
- 如需切换为单线程：
  - 注释掉`main()`中的`with concurrent.futures.ThreadPoolExecutor`部分
  - 取消注释顺序执行代码块

---

## 📄 输出文件说明
### classes.txt

```yaml
nc: 3
names:
  - apple
  - banana
  - orange
```
或者
```yaml
nc: 3
names:
  0: apple
  1: banana
  2: orange
```

### mydata.yaml
```yaml
path: /root/autodl-tmp/datautils/datasets/ 
train: train/images 
val: val/images 
test: test/images 
names: 
  0: apple 
  1: banana 
  2: orange
```

---

## ⚠️ 注意事项
1. JSON文件必须包含以下字段：
   
```json
{
  "imagePath": "img.jpg",
  "imageWidth": 640,
  "imageHeight": 480,
  "shapes": [
    {
      "label": "apple", 
      "shape_type": "rectangle", 
      "points": [
        [x1,y1],
        [x2,y2]
      ] 
    }
  ]
}
```
2. 图像与JSON文件名需保持一致：

```bash
├── img1.jpg 
└── img1.json
```
3. 负样本处理逻辑：
   - 所有无对应JSON的图像会被复制到`images/`目录
   - 不会生成对应的标签文件

4. 文件命名冲突解决：
   - 使用13位数字前缀保证全局唯一性
   - 示例：`0000000000001.jpg`

---

## 📌 常见问题
### Q1: 提示"src_image_path not found"
A: 确保JSON文件中的`imagePath`路径为相对路径，且与实际图像位置匹配

### Q2: 标签顺序不一致
A: 工具会自动按字母序排序生成标签映射，建议保持标签命名统一

### Q3: 转换速度慢
A: 启用多线程模式（默认已启用），或增加系统资源分配


