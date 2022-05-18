
#检测图片
import os
from PIL import Image
from yolo import YOLO

def detect_images(imgs_path):
    for img in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, img)
        image = Image.open(img_path)
        predict_image = yolo.detect_image(image)
        predict_image.show()
        os.system("pause");

if __name__ == '__main__':
    yolo = YOLO()
    imgs_path='./img'
    detect_images(imgs_path)





