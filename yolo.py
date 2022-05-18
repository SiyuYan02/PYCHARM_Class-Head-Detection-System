#-------------------------------------#
#       创建YOLO类
#-------------------------------------#

import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from PIL import Image,ImageFont, ImageDraw

from nets.yolo4 import YoloBody
from utils.config import Config,model_path
from utils.utils import non_max_suppression, DecodeBox,letterbox_image,yolo_correct_boxes


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path": model_path,
        "model_image_size" : (416, 416, 3),
        "confidence": 0.5,
        "iou" : 0.3,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.config = Config
        self.generate()

    #---------------------------------------------------#
    #   初始化
    #---------------------------------------------------#
    def generate(self):
        self.net = YoloBody(3)

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        #将网络预测的结果进行解码
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.config["yolo"]["anchors"][i], (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors loaded.'.format(self.model_path))
        # 生成绘制边框的颜色。
        self.colors = (255,0,0)

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        # ----------------预处理操作-------------------#
        #获取图片高宽
        image_shape = np.array(np.shape(image)[0:2])
        #加上灰条并改变长宽为416*416
        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1])))
        #预处理操作
        photo = np.array(crop_img,dtype = np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        #格式转换成tensor
        images = np.asarray(images)
        images = torch.from_numpy(images)
        if self.cuda:
            images = images.cuda()

        # ----------------feed进入网络-------------------#
        with torch.no_grad():
            # 图片传入网络
            outputs = self.net(images)
            #获取三个维度的输出并解码
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)

            #使用NMS算法去除类别置信度过低的框和IOU重叠过大的框
            batch_detections = non_max_suppression(output, conf_thres=self.confidence, nms_thres=self.iou)

        try :
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image

        #获取置信度和框的大小
        top_conf = batch_detections[:,4]
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(batch_detections[:,0],-1),np.expand_dims(batch_detections[:,1],-1),np.expand_dims(batch_detections[:,2],-1),np.expand_dims(batch_detections[:,3],-1)
        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        #定义字体
        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        #设置目标框线条的宽度
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]
        #利用循环依次画框
        for i in range(len(batch_detections)):
            #获取类别和置信度
            score = top_conf[i]

            #获取框的四个坐标
            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            #防止目标框溢出
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{:.2f}'.format(score)
            draw = ImageDraw.Draw(image)  #创建一个可以在给定图像上绘图的对象
            label_size = draw.textsize(label, font) #标签文字，返回label的宽和高
            label = label.encode('utf-8')
            print(label)

            #确定标签（label）起始点位置：标签的左、下
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors)
            #设置文字框（可注释）
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors)
            # draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image


        # ---------------------------------------------------#
        #   计算mAP使用，返回单张图片预测框的结果
        # ---------------------------------------------------#

    def get_detect_result(self, image):
        self.confidence = 0.5
        self.iou = 0.5
        # ----------------预处理操作-------------------#
        # 获取图片高宽
        image_shape = np.array(np.shape(image)[0:2])
        # 加上灰条并改变长宽为416*416
        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))
        # 预处理操作
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        # 格式转换成tensor
        images = np.asarray(images)
        images = torch.from_numpy(images)
        if self.cuda:
            images = images.cuda()

        # ----------------feed进入网络-------------------#
        with torch.no_grad():
            # 图片传入网络
            outputs = self.net(images)
            #获取三个维度的输出并解码
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)

            #使用NMS算法去除类别置信度过低的框和IOU重叠过大的框
            batch_detections = non_max_suppression(output, conf_thres=self.confidence, nms_thres=self.iou)
        try :
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image

        #获取置信度和框的大小
        top_conf = batch_detections[:,4]
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(batch_detections[:,0],-1),np.expand_dims(batch_detections[:,1],-1),np.expand_dims(batch_detections[:,2],-1),np.expand_dims(batch_detections[:,3],-1)
        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        return top_conf, boxes

