# -*- coding:utf-8 -*-
import os
import numpy as np
import colorsys
import random
from PIL import Image,ImageFont, ImageDraw
from utils.config import class_names,font_path,annotation_path

def show_ground_truth(annotation_line):
    print(annotation_line)
    line = annotation_line.split()
    #获取图片
    image = Image.open(line[0])
    #获取GT_box框
    boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    # 生成绘制边框的颜色。
    # h(色调）：x/len(self.class_names)  s(饱和度）：1.0  v(明亮）：1.0
    # 对于所有种类的目标，确定每一种目标框的绘制颜色，即：将(x/80, 1.0, 1.0)的颜色转换为RGB格式，并随机调整颜色以便于肉眼识别，
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))  # hsv转换为rgb
    # hsv取值范围在[0,1]，而RBG取值范围在[0,255]，所以乘上255
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
    # 定义字体
    font = ImageFont.truetype(font=font_path,size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
    # 设置目标框线条的宽度
    model_image_size=[416,416,3]
    thickness = (np.shape(image)[0] + np.shape(image)[1]) // model_image_size[0]

    for i in range(len(boxes)):
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(image)

        #获取目标框
        left, top, right, bottom, predicted_class = boxes[i]
        predicted_class = class_names[predicted_class]
        label = '{}'.format(predicted_class)
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        # 防止目标框溢出
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

        #定义标签文字
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        # 确定标签（label）起始点位置：标签的左、下
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        #画目标框，线条宽度为thickness(这里画了很多个框重叠起来)
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[class_names.index(predicted_class)])
        #画标签框并填写标签内容（可注释掉）
        # draw.rectangle(
        #     [tuple(text_origin), tuple(text_origin + label_size)],
        #     fill=colors[class_names.index(predicted_class)])
        # draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
    #展示图片
    image.show()


if __name__ == '__main__':
    file = open(annotation_path)
    lines = file.read().splitlines()

    #随机读取一张图片
    #show_ground_truth(random.choice(lines))

    #读取所有图片
    # for line in lines:
    #     show_ground_truth(random.choice(lines))
    #     os.system("pause");