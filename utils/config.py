
# -*- coding:utf-8 -*-

Config = \
{
    "yolo": {
        "anchors": [[[12, 44], [20, 45], [31, 79]],
                    [[6, 22], [9, 20], [14, 30]],
                    [[2, 5], [4, 11], [6, 12]]],
    },
    "img_h": 416,
    "img_w": 416,
}

#字体文件
font_path = './model_data/simhei.ttf'

#图片和xml路径
# image_path = 'G:/数据集/head/JPEGImages/'
# xml_path = 'G:/数据集/head/Annotations/'

#图片和xml路径
image_path = 'G:/dataset/classroom/Images/'
xml_path = 'G:/dataset/classroom/Annotations/'

# #图片和xml路径
# image_path = '/home/imi432_003/pythoncode/dataset/head/JPEGImages/'
# xml_path = '/home/imi432_003/pythoncode/dataset/head/Annotations/'

# #图片和xml路径
# image_path = '/home/imi432_003/pythoncode/dataset/classroom/Images/'
# xml_path = '/home/imi432_003/pythoncode/dataset/classroom/Annotations/'

#存放所有数据集的文件
all_dataset = './model_data/dataset/classroom/dataset.txt'
#生成的训练集的路径
train_dataset = './model_data/dataset/classroom/train.txt'
#生成的测试集的路径
test_dataset = './model_data/dataset/classroom/test.txt'
#权重文件保存结果
model_path = './checkpoints/Epoch458-Total_Loss130.0466-Val_Loss117.8791.pth'

