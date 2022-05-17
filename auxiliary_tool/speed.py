

import numpy as np
import torch
import time
#查看网络结构
from torchsummary import summary

def computeTime(model, device='cuda'):
    inputs = torch.randn(1, 3, 416, 416)

    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in range(100):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Avg execution time (ms): {:.4f}'.format(1/np.mean(time_spent)))




if __name__ == '__main__':
    #卷积加速
    torch.backends.cudnn.benchmark = True

    #加载网络模型
    from nets.yolo4 import YoloBody
    model = YoloBody(3)

    #测试速度
    computeTime(model)
    #查看网络结构
    summary(model, input_size=(3, 416, 416))
