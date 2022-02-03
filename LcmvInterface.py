import os
import sys
import mrcnn.visualize
import mrcnn.model as modellib
from connector import connector
import skimage
from PIL import Image

# 获取该project的根目录
ROOT_DIR = os.path.abspath('')
# 存放训练模型的logs路径
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# 加载预训练的COCO权重
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
sys.path.append(ROOT_DIR)
# datasets文件夹下是训练集和验证集
CONNECTOR_DIR = os.path.join(ROOT_DIR, "datasets")
# 这里输入模型权重的路径
MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, "mask_rcnn_connector_0050.h5")
# 这是输入你要预测图片的路径
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets\\test")


##**********************************************************************
##类名称：LcmvInterface
##类参数：model_dir：模型生成路径  datasets_dir：数据路径
##       model_weight_path：模型权重文件路径  image_dir：预测图片路径
##       epoch：训练循环次数
##类功能：主要提供了模型训练和推断的功能
##**********************************************************************
class LcmvInterface(object):

    def __init__(self, model_dir=MODEL_DIR, datasets_dir=CONNECTOR_DIR,
                model_weight_path=MODEL_WEIGHT_PATH, image_dir=IMAGE_DIR, epoch = 50):
        self.model_dir = model_dir
        self.datasets_dir = datasets_dir
        self.model_weight_path = model_weight_path
        self.image_dir = image_dir
        self.epoch = epoch

    ##**********************************************************************
    ##函数名称：setModelDir
    ##函数参数：model_dir：模型生成路径
    ##返回参数：无
    ##函数功能：设置模型生成路径
    ##**********************************************************************
    def setModelDir(self,model_dir):
        self.model_dir = model_dir

    ##**********************************************************************
    ##函数名称：setDatasetsDir
    ##函数参数：datasets_dir：训练集图片文件夹路径
    ##返回参数：无
    ##函数功能：设置训练集图片文件夹路径
    ##**********************************************************************
    def setDatasetsDir(self, datasets_dir):
        self.datasets_dir = datasets_dir

    ##**********************************************************************
    ##函数名称：setCocoModelPath
    ##函数参数：coco_model_path：coco模型路径
    ##返回参数：无
    ##函数功能：设置coco模型路径
    ##**********************************************************************
    def setCocoModelPath(self, coco_model_path):
        self.coco_model_path = coco_model_path

    ##**********************************************************************
    ##函数名称：setModelWeightPath
    ##函数参数：model_weight_path：权重模型路径
    ##返回参数：无
    ##函数功能：设置权重模型路径
    ##**********************************************************************
    def setModelWeightPath(self, model_weight_path):
        self.model_weight_path = model_weight_path

    ##**********************************************************************
    ##函数名称：setImageDir
    ##函数参数：image_dir：图片路径
    ##返回参数：无
    ##函数功能：设置推测图片路径
    ##**********************************************************************
    def setImageDir(self,image_dir):
        self.image_dir = image_dir

    ##**********************************************************************
    ##函数名称：setEporch
    ##函数参数：eporch：设置训练周期数
    ##返回参数：无
    ##函数功能：设置推测图片路径
    ##**********************************************************************
    def setEporch(self,eporch):
        self.eporch = eporch

    ##**********************************************************************
    ##函数名称：ConnectTrain
    ##函数参数：无
    ##返回参数：无
    ##函数功能：训练模型
    ##**********************************************************************
    def ConnectTrain(self):
        # 从balloon文件中引入配置
        config = connector.ConnectorConfig()

        # 加载训练集
        dataset_train = connector.ConnectorDataset()
        dataset_train.load_connector(self.datasets_dir, "train")
        dataset_train.prepare()

        # 加载验证集
        dataset_val = connector.ConnectorDataset()
        dataset_val.load_connector(self.datasets_dir, "val")
        dataset_val.prepare()

        # 判断当前路径中是否存在当前路径的文件夹，若不存在就创建
        mkdir(self.model_dir)

        # 创建一个训练模型
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=self.model_dir)

        # 设定权重的初始化方式，有imagenet，coco,last三种,选择coco
        init_with = "coco"

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)

        elif init_with == "coco":
            model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                                                       "mrcnn_bbox", "mrcnn_mask"])

        elif init_with == "last":
            model.load_weights(model.find_last()[1], by_name=True)

        # 设定训练参数，如学习率，epoch等
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=self.epoch, layers="all")

    ##**********************************************************************
    ##函数名称：ConnectModelLoad
    ##函数参数：无
    ##返回参数：model：加载权重后的模型
    ##函数功能：加载模型
    ##**********************************************************************
    def ConnectModelLoad(self):
        # 创建一个推测配置类，其继承ConnectorConfig类
        class InferenceConfig(connector.ConnectorConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        # 实类化一个推测类对象
        inferenceConfig = InferenceConfig()
        # 创建一个MASK-RCNN的预测对象
        model = modellib.MaskRCNN(mode="inference", model_dir=self.model_dir, config=inferenceConfig)
        # 获取权重路径
        dirs = os.listdir(self.model_dir) # 获取模型文件夹下的列表
        mydir = dirs[len(dirs)-1]         # 获取最新训练的模型文件夹
        # 遍历获取最新文件夹下的权重文件，并且保存在
        for root, dirs, files in os.walk(os.path.join(self.model_dir,mydir)):
            for file in files:
                if file.endswith(str(self.epoch)+".h5"):
                    self.model_weight_path = os.path.join(os.path.join(self.model_dir,mydir), file)
        # 加载权重
        model.load_weights(self.model_weight_path, by_name=True)
        # 返回该模型
        return model

    ##**********************************************************************
    ##函数名称：ConnectInference
    ##函数参数：model：加载权重后的模型
    ##返回参数：无
    ##函数功能：根据模型推断
    ##**********************************************************************
    def ConnectInference(self, model):
        changePictureSize(self.image_dir, 512, 512)
        # 添加自己的分类的名称
        class_names = ['BG', 'badConnector','goodConnector']
        # 将测试集中的图片名存放到file_names中
        file_names = next(os.walk(self.image_dir))[2]
        # 循环读取图片文件夹中的图片
        # 按照[[类名,x1,y1,x2,y2],[...],...]的方式进行字符串拼接，并且返回
        mystr = '['
        for x in range(len(file_names)):
            image = skimage.io.imread(os.path.join(self.image_dir, file_names[x]))
            # 执行结果预测
            results = model.detect([image], verbose=1)
            # 可视化输出
            r = results[0]
            # 显示出测试的图片
            mrcnn.visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
            # 按照[[类名,x1,y1,x2,y2],[...],...]的方式进行字符串拼接，并且返回
            mystr = mystr + '['
            for y in range(len(r['class_ids'])):
                x1 = r['rois'][y][0]
                y1 = r['rois'][y][1]
                x2 = r['rois'][y][2]
                y2 = r['rois'][y][3]
                if y == len(r['class_ids'])-1:
                    mystr = mystr + '[' + str(class_names[r['class_ids'][y]])+ ',' + str(x1) + ',' + str(
                        y1) + ',' + str(x2) + ',' + str(y2) + ']'
                else:
                    mystr = mystr + '[' + str(class_names[r['class_ids'][y]]) + ',' + str(x1) + ',' + str(
                        y1) + ',' + str(x2) + ',' + str(y2) + ']' + ','
            if x == len(file_names)-1:
                mystr = mystr + ']'
            else:
                mystr = mystr + '], '
        mystr = mystr + ']'
        print(mystr)
        return mystr

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
            if i.endswith('.jpg'):
                im = Image.open(os.path.join(image_dir,i))
                out = im.resize((height, width),Image.ANTIALIAS)
                out.save(os.path.join(image_dir,i))

##**********************************************************************
##函数名称：mkdir
##函数参数：path：文件路径
##返回参数：无
##函数功能：判断当前路径下该文件夹是否存在，若不存在就创建
##**********************************************************************
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

if __name__=='__main__':
    # changePictureSize("C:\\Users\\ZYL\\Desktop\\浇口高\\",512,512)
    lc = LcmvInterface()
    lc.ConnectTrain()
