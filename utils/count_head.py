#
# #读取图片和xml文件并将其信息写入到txt中
# import os
# import random
# import xml.etree.ElementTree as ET
#
# from utils.config import image_path,xml_path,all_dataset,train_dataset,test_dataset
#
# #读取xml文件得到所有标注框
# def convert_annotation(image_id, list_file):
#     # rb:二进制文件
#     in_file = open(os.path.join(xml_path,'%s.xml'%(image_id)),"rb")
#     tree=ET.parse(in_file)
#     root = tree.getroot()
#
#     for obj in root.iter('object'):
#
#         #读取难识别标签
#         read_difficult=False
#         if read_difficult:
#             if obj.find('difficult') != None:
#                 difficult = obj.find('difficult').text
#             if int(difficult) == 1:
#                 continue
#
#         xmlbox = obj.find('bndbox')
#         b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
#         list_file.write(" " + ",".join([str(a) for a in b]) )
#     list_file.write('\n')
#
# if __name__ == '__main__':
#     #读取所有数据集
#     list_file = open(all_dataset, 'w')
#     for file in os.listdir(image_path):
#         image_id = os.path.splitext(file)[0] #获取文件名
#         list_file.write(os.path.join(image_path, file)) #写图片完整路径
#         convert_annotation(image_id, list_file) #写标注框
#     list_file.close()
#
# # -*- coding:utf-8 -*-

# import os
#
# import xml.dom.minidom
#
# xml_file_path = "/home/lyz/data/VOCdevkit/MyDataSet/Annotations/"
# lst_label = ["height", "width", "depth"]
# lst_dir = os.listdir(xml_file_path)
#
#
# for file_name in lst_dir:
#     file_path = xml_file_path + file_name
#     tree = xml.dom.minidom.parse(file_path)
#     root = tree.documentElement		#获取根结点
#     size_node = root.getElementsByTagName("size")[0]
#     for size_label in lst_label:	#替换size标签下的子节点
#         child_tag = "img_" + size_label
#         child_node = size_node.getElementsByTagName(child_tag)[0]
#         new_node = tree.createElement(size_label)
#         text = tree.createTextNode(child_node.firstChild.data)
#         new_node.appendChild(text)
#         size_node.replaceChild(new_node, child_node)
#
#     #替换object下的boundingbox节点
#     lst_obj = root.getElementsByTagName("object")
#     data = {}
#     for obj_node in lst_obj:
#         box_node = obj_node.getElementsByTagName("bounding_box")[0]
#         new_box_node = tree.createElement("bndbox")
#         for child_node in box_node.childNodes:
#             tmp_node = child_node.cloneNode("deep")
#             new_box_node.appendChild(tmp_node)
#         x_node = new_box_node.getElementsByTagName("x_left_top")[0]
#         xmin = x_node.firstChild.data
#         data["xmin"] = (xmin, x_node)
#         y_node = new_box_node.getElementsByTagName("y_left_top")[0]
#         ymin = y_node.firstChild.data
#         data["ymin"] = (ymin, y_node)
#         w_node = new_box_node.getElementsByTagName("width")[0]
#         xmax = str(int(xmin) + int(w_node.firstChild.data))
#         data["xmax"] = (xmax, w_node)
#         h_node = new_box_node.getElementsByTagName("height")[0]
#         ymax = str(int(ymin) + int(h_node.firstChild.data))
#         data["ymax"] = (ymax, h_node)
#
#
#         for k, v in data.items():
#             new_node = tree.createElement(k)
#             text = tree.createTextNode(v[0])
#             new_node.appendChild(text)
#             new_box_node.replaceChild(new_node, v[1])
#         obj_node.replaceChild(new_box_node, box_node)
#
#     with open(file_path, 'w') as f:
#         tree.writexml(f, indent="\n", addindent="\t", encoding='utf-8')
#
#     #去掉XML文件头（一些情况下文件头的存在可能导致错误）
#     lines = []
#     with open(file_path, 'rb') as f:
#         lines = f.readlines()[1:]
#     with open(file_path, 'wb') as f:
#         f.writelines(lines)
#
# print("-----------------done--------------------")



import xml.dom.minidom
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

count_headnum = [0 for i in range(3070)]
count_headarea = [0 for i in range(106304)]

i = 0
sum = 0
#存放xml文件的地址
xml_file_path = r"D:/python_files/classroom/Annotations/"
lst_dir = os.listdir(xml_file_path)

for file_name in lst_dir:

    #读入所有的xml文件
    file_path = xml_file_path + file_name
    tree = xml.dom.minidom.parse(file_path)
    #获取根节点
    root = tree.documentElement
    persons = root.getElementsByTagName("object")

    for person in persons:

        count_headnum[i] = count_headnum[i] + 1

        head_area = person.getElementsByTagName("bndbox")[0]
        x_min = head_area.getElementsByTagName("xmin")[0].firstChild.data
        x_max = head_area.getElementsByTagName("xmax")[0].firstChild.data
        y_min = head_area.getElementsByTagName("ymin")[0].firstChild.data
        y_max = head_area.getElementsByTagName("ymax")[0].firstChild.data
        count_headarea[j] = (int(x_max) - int(x_min))*(int(y_max) - int(y_min))

        j = j + 1

    i = i + 1

# df.sample(frac=0.8, replace=True, random_state=1)

from sklearn.model_selection import train_test_split
train_test, validation_data = train_test_split(count_headnum, train_size=0.684, test_size=0.316)

train_data,test_data = train_test_spilt(train_test,train = 0.167,test_size = 0.833)

train_test_area, validation__area = train_test_split(count_headarea, train_size=0.684, test_size=0.316)

train__area,test__area = train_test_spilt(train_test_area,train = 0.167,test_size = 0.833)

print(count_headnum)
# print(train_data)
# print(validation_data)
# print(test_data)

print(count_headarea)
# print(train_area)
# print(validation_area)
# print(test_area)

