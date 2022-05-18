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
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

count_headnum = [0 for i in range(3070)]
count_headarea = [0 for i in range(106304)]
count_headnum = [0 for i in range(3070)]
x_min = [0 for i in range(106304)]
y_min = [0 for i in range(106304)]
x_max = [0 for i in range(106304)]
y_max = [0 for i in range(106304)]

i = 0
sum = 0
j = 0

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
        x_min[j] = head_area.getElementsByTagName("xmin")[0].firstChild.data
        x_max[j] = head_area.getElementsByTagName("xmax")[0].firstChild.data
        y_min[j] = head_area.getElementsByTagName("ymin")[0].firstChild.data
        y_max[j] = head_area.getElementsByTagName("ymax")[0].firstChild.data
        count_headarea[j] = (int(x_max[j]) - int(x_min[j]))*(int(y_max[j]) - int(y_min[j]))

        j = j + 1

    i = i + 1

# print(count_headnum.shape)
# random.shuffle(count_headnum)  # 打乱列表顺序
#
# train = count_headnum[:1799]
# val = count_headnum[1800:2099]
# test = count_headnum[2100:]


# def count_(train,count_train):
#     count_train = [0 for i in range(11)]
#     for i in range(len(train)):
#         if train[i]>=0 and train[i]<10 :
#             count_train[0] = count_train[0] + 1
#         if train[i] >= 10 and train[i] < 20:
#             count_train[1] = count_train[1] + 1
#         if train[i] >= 20 and train[i] < 30:
#             count_train[2] = count_train[2] + 1
#         if train[i] >= 30 and train[i] < 40:
#             count_train[3] = count_train[3] + 1
#         if train[i] >= 40 and train[i] < 50:
#             count_train[4] = count_train[4] + 1
#         if train[i] >= 50 and train[i] < 60:
#             count_train[5] = count_train[5] + 1
#         if train[i] >= 60 and train[i] < 70:
#             count_train[6] = count_train[6] + 1
#         if train[i] >= 70 and train[i] < 80:
#             count_train[7] = count_train[7] + 1
#         if train[i] >= 80 and train[i] < 90:
#             count_train[8] = count_train[8] + 1
#         if train[i] >= 90 and train[i] < 100:
#             count_train[9] = count_train[9] + 1
#         if train[i] >= 100 and train[i] < 110:
#             count_train[10] = count_train[10] + 1
#     return count_train

# count_train = [0 for i in range(11)]
# count_train = count_(train,count_train)
#
# count_test = [0 for i in range(11)]
# count_test = count_(test,count_test)
#
# count_val = [0 for i in range(11)]
# count_val = count_(val,count_val)
#
# print(count_train)
# print(count_test)
# print(count_val)
#
# def count_area(train,count_train):
#     count_train = [0 for i in range(6)]
#     for i in range(len(train)):
#         if train[i]>=0 and train[i]<200 :
#             count_train[0] = count_train[0] + 1
#         if train[i] >= 200 and train[i] < 400:
#             count_train[1] = count_train[1] + 1
#         if train[i] >= 400 and train[i] < 600:
#             count_train[2] = count_train[2] + 1
#         if train[i] >= 600 and train[i] < 800:
#             count_train[3] = count_train[3] + 1
#         if train[i] >= 800 and train[i] < 1000:
#             count_train[4] = count_train[4] + 1
#         if train[i] >= 1000 :
#             count_train[5] = count_train[5] + 1
#         # if train[i] >= 800 and train[i] < 1000:
#         #     count_train[6] = count_train[6] + 1
#         # if train[i] >= 1000 and train[i] < 2000:
#         #     count_train[7] = count_train[7] + 1
#         # if train[i] >= 2000 and train[i] < 5000:
#         #     count_train[8] = count_train[8] + 1
#         # if train[i] >= 5000 and train[i] < 10000:
#         #     count_train[9] = count_train[9] + 1
#         # if train[i] >= 10000 :
#         #     count_train[10] = count_train[10] + 1
#     return count_train
#
# random.shuffle(count_headarea)  # 打乱列表顺序
#
# rate = 106374/3070
# print(rate)
#
# train_area = count_headarea[:round(1799*rate)]
# val_area = count_headarea[round(1800*rate):round(2099*rate)]
# test_area = count_headarea[round(2100*rate):]
#
# count_train = [0 for i in range(6)]
# count_train = count_area(train_area,count_train)
# #
# count_test = [0 for i in range(6)]
# count_test = count_area(test_area,count_test)
#
# count_val = [0 for i in range(6)]
# count_val = count_area(val_area,count_val)
#
# print(count_train)
# print(count_test)
# print(count_val)

mat = np.array(count_headarea)
print(np.argmin(mat))
print(x_min[np.argmin(mat)])
print(y_min[np.argmin(mat)])
print(x_max[np.argmin(mat)])
print(y_max[np.argmin(mat)])
