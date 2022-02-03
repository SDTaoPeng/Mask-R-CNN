
import os
import json
import numpy as np
from skimage import io,draw
from mrcnn.config import Config
from mrcnn import utils

class LungDataset(utils.Dataset):

    def load_lung(self, dataset_dir, subset):
        """定义数据集有哪些类别.
        dataset_dir: 数据集根路径.
        subset: train or val
        """
        # 添加类别的名称和ID号，在这里，为了简便，我只添加一类：‘connector’
        self.add_class("lung", 1, "right")
        self.add_class("lung", 2, "left")
        # 加载训练集还是测试集
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # 读入json标注数据集
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))

        # 获得字典的键值
        annotations = list(annotations.values())

        # 拿到每一个regions的信息并组成一个列表
        annotations = [a for a in annotations if a['regions']]

        # 加载图片
        for a in annotations:
            # 获取组成每个物体实例轮廓的多边形点的x、y坐标。
            # 这些坐标保存在r['shape_attributes'中，（参见上面的json格式）
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            temp = [r['region_attributes'] for r in a['regions'].values()]
            #print(temp)
            temp = [r['name'] for r in temp]
            # print(temp)
            # 序列字典,可以在此处添加其他类，要求与前面的add_class一致
            # name_dict = {"ng": 1,"ok":2,"discolor":3}
            name_dict = {"right": 1,"left":2}
            name_id = [name_dict[a] for a in temp]
            # 根据图片名称获取图片存储路径
            image_path = os.path.join(dataset_dir, a['filename'])
            image = io.imread(image_path)
            # 读取图片中的高度和宽度
            height, width = image.shape[:2]
            self.add_image(
                "lung",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                class_id=name_id,
                polygons=polygons)

    def load_mask(self, image_id):
        """为图像生成实例mask.
       Returns:
        masks:  一个bool数组，每个实例一个mask,
                其中每个mask为：[高, 宽, 数目]
        class_ids: 每个mask对应的类别.
        """
        info = self.image_info[image_id]
        if info["source"] != "lung":
            return super(self.__class__, self).load_mask(image_id)

        # 将多边形转换为一个二值mask,[高, 宽, 通道数目]
        name_id = info["class_id"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        class_ids = np.array(name_id, dtype=np.int32)
        for i, p in enumerate(info["polygons"]):
            # 获取多边形内的像素索引并将其设置为1
            rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # 返回mask和类别的信息
        return mask.astype(np.bool), class_ids


    def image_reference(self, image_id):
        # 返回为图片存储路径
        info = self.image_info[image_id]

        if info["source"] == "connector":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class LungConfig(Config):

    # 给配置类一个名称
    NAME = "lung"
    # GPU数量
    GPU_COUNT = 1
    # 根据自己的GPU进行配置
    IMAGES_PER_GPU = 1
    # 类的总数，包括背景
    NUM_CLASSES = 1 + 2  # Background + connector
    # 每次训练所需的步数
    STEPS_PER_EPOCH = 100
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7
    # 学习率
    LEARNING_RATE = 0.001
    # backBone的选择 "resnet50" 或者 "resnet101",为图片特征提取主干网络
    BACKBONE = "resnet50"
    # 权重下降梯度
    WEIGHT_DECAY = 0.0001
    # 图片最大最小像素
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

