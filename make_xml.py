

# 导入minidom
import xml.dom.minidom as DOC
import os
from yolo import YOLO
from PIL import Image

# 将bounding box信息写入xml文件中, bouding box格式为[[x_min, y_min, x_max, y_max, name]]
def generate_xml(img_name, bboxes, save_xml_path, img_size, depth=3):
    '''
    输入：
        img_name：图片名称，如a.jpg
        bboxes:坐标list，格式为[[x_min, y_min, x_max, y_max]]
        img_size：图像的大小,格式为[h,w]
        save_xml_path: xml文件输出的根路径
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
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(depth))
    title.appendChild(title_text)
    size.appendChild(title)

    for box in bboxes:
        object = doc.createElement('object')
        annotation.appendChild(object)

        #设置类别名称
        title = doc.createElement('name')
        title_text = doc.createTextNode('person')
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
        title_text = doc.createTextNode(str(int(float(box[1]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(float(box[0]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(float(box[3]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(float(box[2]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open(os.path.join(save_xml_path, img_name.split('.')[0] + '.xml'), 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


if __name__ == '__main__':
    yolo = YOLO()
    image_path = 'G:/test_img/img'
    save_xml_path = 'G:/test_img/xml'

    imgs = os.listdir(image_path)
    count = 0
    
    for img_name in imgs:
        img_path = os.path.join(image_path, img_name)

        image = Image.open(img_path).convert('RGB')
        try:
            top_conf, bboxes = yolo.get_detect_result(image)
        except:
            continue


        depth = 3

        generate_xml(img_name, bboxes, save_xml_path, list(image.size), depth)

        count += 1
        print(count, '/', len(imgs))


