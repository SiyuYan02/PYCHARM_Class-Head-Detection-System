
#计算mAP的辅助函数
import os
from PIL import Image
import numpy as np
from yolo import YOLO
from utils.config import test_dataset

#设置类别名称
label='person'
#设置文件保存地址
gt_save_path = 'auxiliary_tool/mAP/input/ground-truth/'
predict_save_path = 'auxiliary_tool/mAP/input/detection-results/'

#读取标签
def create_ground_truth(gt_path, save_path):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            file_name = (line[0].split('/')[-1]).split('.')[0]
            with open(save_path + file_name + '.txt', 'w', encoding='utf-8') as t:
                boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
                for box in boxes:
                    t.write('%s %s %s %s %s\n' % (label, box[0], box[1], box[2], box[3]))
            t.close()
    f.close()

#生成预测结果
def create_predict(test_path, save_path):
    yolo = YOLO()
    #读取测试数据集
    with open(test_path) as f:
        lines = f.readlines()
    f.close()

    for annotation_line in lines:
        line = annotation_line.split()
        image_name = (line[0].split('/')[-1]).split('.')[0]
        image = Image.open(line[0])
        try:
            conf, boxes = yolo.get_detect_result(image)
            with open(os.path.join(save_path, image_name) + '.txt', 'w', encoding='utf-8') as t:
                for i in range(len(conf)):
                    t.write('%s %s %d %d %d %d\n' % (label, str(conf[i]), int(boxes[i][1]), int(boxes[i][0]), int(boxes[i][3]), int(boxes[i][2])))
            t.close()
        except:
            with open(os.path.join(save_path, image_name) + '.txt', 'w', encoding='utf-8') as t:
                pass

if __name__ == '__main__':
    create_ground_truth(test_dataset,gt_save_path)
    create_predict(test_dataset,predict_save_path)

