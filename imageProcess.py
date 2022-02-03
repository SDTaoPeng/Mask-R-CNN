from PIL import Image,ImageEnhance
import os

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
                img = Image.open(os.path.join(image_dir,i))
                # 亮化将背景处理成白色
                enh_bri = ImageEnhance.Brightness(img)
                brightness = 3
                img_brightness = enh_bri.enhance(brightness)
                # 锐化
                enh_sha = ImageEnhance.Sharpness(img_brightness)
                sharpness = 3.0
                img_sharp = enh_sha.enhance(sharpness)
                #
                out = img_sharp.resize((height, width),Image.ANTIALIAS)
                out.save(os.path.join(image_dir,i))


def changePictureSize(image_dir, width, height):
    count = 0
    for root, dirs, files in os.walk(image_dir):
        for i in files:
            if i.endswith('.jpg'):
                im = Image.open(os.path.join(image_dir,i))
                out = im.resize((height, width),Image.ANTIALIAS)
                out.save(os.path.join(image_dir,os.path.join("C:\\Users\\ZYL\\Desktop\\demo5"),"blot_"+str(count)+".jpg"))
                count = count+1

if __name__=='__main__':
    changePictureSize("C:\\Users\\ZYL\\Desktop\\blot",128,256)