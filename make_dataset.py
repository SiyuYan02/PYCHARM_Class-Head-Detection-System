
#读取图片和xml文件并将其信息写入到txt中
import os
import random
import xml.etree.ElementTree as ET

from utils.config import image_path,xml_path,all_dataset,train_dataset,test_dataset

#读取xml文件得到所有标注框
def convert_annotation(image_id, list_file):
    in_file = open(os.path.join(xml_path,'%s.xml'%(image_id)),"rb")
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):

        #读取难识别标签
        read_difficult=False
        if read_difficult:
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
            if int(difficult) == 1:
                continue

        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) )
    list_file.write('\n')

if __name__ == '__main__':
    #读取所有数据集
    list_file = open(all_dataset, 'w')
    for file in os.listdir(image_path):
        image_id = os.path.splitext(file)[0] #获取文件名
        list_file.write(os.path.join(image_path, file)) #写图片完整路径
        convert_annotation(image_id, list_file) #写标注框
    list_file.close()

    #一定比例分割数据集
    total_image = os.listdir(image_path)
    ratio = 0.8
    num = int(len(total_image) * ratio)
    random.shuffle(total_image)
    train = total_image[:num]
    test = total_image[num:]
    #将训练集和测试集写入txt
    list_file = open(train_dataset, 'w')
    for image in train:
        list_file.write(os.path.join(image_path, image))  # 写图片完整路径
        image_id = os.path.splitext(image)[0]  # 获取文件名
        convert_annotation(image_id, list_file)  # 写标注框
    list_file.close()

    list_file = open(test_dataset, 'w')
    for image in test:
        list_file.write(os.path.join(image_path, image))  # 写图片完整路径
        image_id = os.path.splitext(image)[0]  # 获取文件名
        convert_annotation(image_id, list_file)  # 写标注框
    list_file.close()
