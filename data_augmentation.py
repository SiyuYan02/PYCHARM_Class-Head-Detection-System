# -*- coding=utf-8 -*-
import os
import random
import cv2
import numpy as np
from skimage import exposure
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOC

# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(one_xml_path):
    '''
    输入：
        xml_path: xml的文件路径
    输出：
        从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    '''
    tree = ET.parse(one_xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords


# 将bounding box信息写入xml文件中, bouding box格式为[[x_min, y_min, x_max, y_max, name]]
def generate_xml(img_name, coords, img_size, out_root_path, cnt):
    '''
    输入：
        img_name：图片名称，如a.jpg
        coords:坐标list，格式为[[x_min, y_min, x_max, y_max, name]]，name为概况的标注
        img_size：图像的大小,格式为[h,w,c]
        out_root_path: xml文件输出的根路径
    '''
    doc = DOC.Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('classroom')
    title.appendChild(title_text)
    annotation.appendChild(title)

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The classroom Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('classroom')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for coord in coords:
        object = doc.createElement('object')
        annotation.appendChild(object)

        title = doc.createElement('name')
        title_text = doc.createTextNode(coord[4])
        title.appendChild(title_text)
        object.appendChild(title)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        object.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        title = doc.createElement('xmin')
        title_text = doc.createTextNode(str(int(float(coord[0]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(float(coord[1]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(float(coord[2]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(float(coord[3]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open(os.path.join(out_root_path, "argumentation_" + cnt + '.xml'), 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
        cv2.putText(img, bbox[4], (int(x_min), int(y_min)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), thickness=2)
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.remove('./1.jpg')


# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, flip_rate = 1, add_noise_rate=0, change_light_rate=0.8, crop_rate=1):
        self.flip_rate = flip_rate
        self.add_noise_rate = add_noise_rate
        self.change_light_rate = change_light_rate
        self.crop_rate = crop_rate

    # 翻转
    def _flip_pic_bboxes(self, img, bboxes):
        w = img.shape[1]
        flip_img = np.fliplr(img)

        flip_bboxes = np.array(bboxes)
        flip_bboxes[:, [0, 2]] = w - flip_bboxes[:, [2, 0]]
        flip_bboxes = flip_bboxes.tolist()

        return flip_img, flip_bboxes

    # 高斯模糊
    def _addNoise(self, img):
        size = random.choice((3,5,7))
        return cv2.GaussianBlur(img, ksize=(size, size), sigmaX=0, sigmaY=0)

    # 调整亮度
    def _changeLight(self, img):
        flag = random.uniform(0.7, 1.3)  # flag>1为调暗,小于1为调亮
        return exposure.adjust_gamma(img, flag)

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes


    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        if len(bboxes) == 0:
            return img, []

        print('------')

        if random.random() < self.flip_rate:  # 翻转
            print('翻转')
            img, bboxes = self._flip_pic_bboxes(img, bboxes)

        if random.random() < self.add_noise_rate:  # 加噪声
            print('高斯模糊')
            img = self._addNoise(img)

        if random.random() < self.change_light_rate:  # 改变亮度
            print('亮度')
            img = self._changeLight(img)

        if random.random() < self.crop_rate:  # 裁剪
            print('裁剪')
            img, bboxes = self._crop_img_bboxes(img, bboxes)

        print('\n')

        return img, bboxes


if __name__ == '__main__':

    dataAug = DataAugmentForObjectDetection()

    save_img_path = "G:/dataset/classroom/Images"
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    save_xml_path = "G:/dataset/classroom/Annotations"
    if not os.path.exists(save_xml_path):
        os.makedirs(save_xml_path)

    image_path = "F:/classroom/Images"
    xml_path = "F:/classroom/Annotations"

    for file in os.listdir(image_path):

        image_id = os.path.splitext(file)[0]  # 获取文件名
        one_image_path = os.path.join(image_path, file)  # 图片完整路径
        one_xml_path = os.path.join(xml_path,'%s.xml'%(image_id)) #标注文件路径

        print(image_id)

        img = cv2.imread(one_image_path) #读取图片
        coords = parse_xml(one_xml_path) #读取xml文件

        names = [coord[4] for coord in coords]
        coords = [coord[:4] for coord in coords]
        auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
        [auged_bboxes[i].append(name) for i, name in enumerate(names)]

        #show_pic(auged_img, auged_bboxes)  # 强化后的图
        #保存图片和xml文件
        cv2.imwrite(os.path.join(save_img_path, "argumentation_" + file), auged_img)
        generate_xml(file, auged_bboxes, list(auged_img.shape), save_xml_path, image_id)

        #os.system("pause")



