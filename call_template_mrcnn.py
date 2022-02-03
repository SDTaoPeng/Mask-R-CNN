import os
import sys
import mrcnn.visualize
import mrcnn.model as modellib
import mrcnn.config as Config
from lung import lung
import skimage
from PIL import Image,ImageEnhance
import numpy as np
import cv2

# 获取该project的根目录
ROOT_DIR = os.path.abspath('')
# 存放训练模型的logs路径
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
sys.path.append(ROOT_DIR)
# 这里输入模型权重的路径
MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, "mask_rcnn_lung_0030.h5")
# 这是输入你要预测图片的路径
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets\\test")

##**********************************************************************
##函数名称：run_model
##函数参数：model_dir：模型路径，默认路径为MODEL_DIR。
#          model_weight_path：模型权重路径，默认路径为MODEL_WEIGHT_PATH。
##返回参数：model：加载权重后的模型,默认路径为MODEL_DIR，即工程目录下的logs。
##函数功能：模型的权重进行加载。
##**********************************************************************
def run_model(model_dir=MODEL_DIR, model_weight_path=MODEL_WEIGHT_PATH):

    # 创建一个推测配置类，其继承ConnectorConfig类
    class InferenceConfig(lung.LungConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        STEPS_PER_EPOCH = 10


    # 实类化一个推测类对象
    inferenceConfig = InferenceConfig()
    # 创建一个MASK-RCNN的预测对象
    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=inferenceConfig)
    # 加载权重
    model.load_weights(model_weight_path, by_name=True)
    # 返回该模型
    return model

##**********************************************************************
##函数名称：inference_mrcnn
##函数参数：model：权重加载后的模型
#          image_dir：待检测图片路径。
##返回参数：paras：是一个list，按照[[category,vertex[0,0],...]的方式
##函数功能：对目录下的图片进行预测。
##**********************************************************************
def inference_mrcnn(model, image_dir=IMAGE_DIR):
    # 将路径下的图片改换成512*512的
    changePictureSize(image_dir, 512, 512)
    # 添加自己的分类的名称
    # class_names = ['BG', 'ng','ok','discolor']
    # class_names = ['BG', 'tooHigh','imcomplete','blot']
    class_names = ['BG','right','left']
    # 将测试集中的图片名存放到file_names中
    file_names = next(os.walk(image_dir))[2]
    # 循环读取图片文件夹中的图片
    # 按照[[类名,x1,y1,x2,y2],[...],...]的方式进行字符串拼接，并且返回
    paras = []
    for x in range(len(file_names)):
        image = skimage.io.imread(os.path.join(image_dir, file_names[x]))
        results = model.detect([image], verbose=1)
        # 可视化输出
        r = results[0]
        print(r['class_ids'])
        # 显示出测试的图片
        mrcnn.visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        # 按照[[category,vertex[0,0],...]的方式进行字符串拼接，并且返回
        if len(r['masks']) > 0:
            para = []
            for i in range(len(r['rois'])):
                vertex = []
                para.append(r['class_ids'][i])
                x1 = r['rois'][i][0]
                y1 = r['rois'][i][1]
                x2 = r['rois'][i][2]
                y2 = r['rois'][i][3]
                for x in range(x1,x2):
                    for y in range(y1,y2):
                        if (r['masks'][x][y][0] != 0):
                            vertex.append(x)
                            vertex.append(y)
                para.append(vertex)
            print(str(para))
            # for roi in r['rois']:
            #     vertex = []
            #     for r in roi:
            #         vertex.append(r)
            #     para.append(vertex)
            paras.append(para)
    print(str(paras))
    return paras

##**********************************************************************
##函数名称：changePictureSize
##函数参数：image_dir：图片文件夹路径
##         width：图片修改后的宽度
##         height：图片修改后的高度
##返回参数：无
##函数功能：将目标文件夹下的图片修改为指定宽高的图片
##**********************************************************************
def changePictureSize(image_dir, width, height):
    for root, dirs, files in os.walk(image_dir):
        for i in files:
            if i.endswith('.png'):
                im = Image.open(os.path.join(image_dir, i))
                out = im.resize((height, width), Image.ANTIALIAS)
                out.save(os.path.join(image_dir, i))

##**********************************************************************
##函数名称：shapePicture
##函数参数：imagePath：图片文件夹路径
##返回参数：img_sharp：锐化后的图片
##函数功能：将图片进行锐化处理，使得边界清晰
##**********************************************************************
def shapePicture(imagePath):
    img = Image.open(imagePath)
    # 亮化将背景处理成白色
    # enh_bri = ImageEnhance.Brightness(img)
    # brightness = 3
    # img_brightness = enh_bri.enhance(brightness)
    # 锐化
    enh_sha = ImageEnhance.Sharpness(img)
    sharpness = 2.0
    img_sharp = enh_sha.enhance(sharpness)
    return img_sharp

##**********************************************************************
##函数名称：brightPicture
##函数参数：imagePath：图片文件夹路径
##返回参数：img_sharp：亮化后的图片
##函数功能：将图片量化处理
##**********************************************************************
def brightPicture(imagePath):
    img = Image.open(imagePath)
    # 亮化将背景处理成白色
    enh_bri = ImageEnhance.Brightness(img)
    brightness = 3
    img_brightness = enh_bri.enhance(brightness)
    return img_brightness

if __name__=='__main__':
    model = run_model()
    inference_mrcnn(model)