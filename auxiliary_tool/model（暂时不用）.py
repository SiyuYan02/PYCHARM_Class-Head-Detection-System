
import torch.nn as nn
import torch

#卷积模块，包括卷积层、batch norm(BN)层、LeakyReLU激活函数
class DarknetConv2D_BN_Leaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1):
        super(DarknetConv2D_BN_Leaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyReLU(x)
        return x

#残差模块,包括输入通道数，输出通道数，残差模块的堆叠次数
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_Block):
        super(ResidualBlock, self).__init__()
        self.num_Block = num_Block
        self.dark_conv1 = DarknetConv2D_BN_Leaky(in_channels, out_channels, ksize = 3, stride = 2, padding = 1)
        self.dark_conv2 = []
        for i in range(self.numBlock):
            layers = []
            layers.append(DarknetConv2D_BN_Leaky(out_channels, out_channels//2, ksize = 1, stride = 1, padding = 0))
            layers.append(DarknetConv2D_BN_Leaky(out_channels//2, out_channels, ksize = 3, stride = 1, padding = 1))
            self.dark_conv2.append(nn.Sequential(*layers))
        self.dark_conv2 = nn.ModuleList(self.dark_conv2)
    def forward(self, x):
        x = self.dark_conv1(x)
        for convblock in self.dark_conv2:
            residual = x
            x = self.convblock(x)
            x = x + residual
        return x

#后端输出模块
class LastLayer(nn.Module):
    def __init__(self, numIn, numOut, numOut2):
        super(LastLayer, self).__init__()
        self.dark_conv1 = DarknetConv2D_BN_Leaky(numIn, numOut, ksize=1, stride=1, padding=0)
        self.dark_conv2 = DarknetConv2D_BN_Leaky(numOut, numOut * 2, ksize=3, stride=1, padding=1)
        self.dark_conv3 = DarknetConv2D_BN_Leaky(numOut * 2, numOut, ksize=1, stride=1, padding=0)
        self.dark_conv4 = DarknetConv2D_BN_Leaky(numOut, numOut * 2, ksize=3, stride=1, padding=1)
        self.dark_conv5 = DarknetConv2D_BN_Leaky(numOut * 2, numOut, ksize=1, stride=1, padding=0)

        self.dark_conv6 = DarknetConv2D_BN_Leaky(numOut, numOut * 2, ksize=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(numOut * 2, numOut2, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.dark_conv1(x)
        x = self.dark_conv2(x)
        x = self.dark_conv3(x)
        x = self.dark_conv4(x)
        x = self.dark_conv5(x)

        y = self.dark_conv6(x)
        y = self.conv7(y)
        return x, y


class Yolov3(nn.Module):
    def __init__(self, numAnchor, numClass):
        super(Yolov3, self).__init__()
        self.dark_conv1 = DarknetConv2D_BN_Leaky(3, 32, ksize=3, stride=1, padding=1)
        self.res1 = ResidualBlock(32, 64, 1)
        self.res2 = ResidualBlock(64, 128, 2)
        self.res3 = ResidualBlock(128, 256, 8)
        self.res4 = ResidualBlock(256, 512, 8)
        self.res5 = ResidualBlock(512, 1024, 4)

        self.last1 = LastLayer(1024, 512, numAnchor * (numClass + 5))
        self.up1 = nn.Sequential(DarknetConv2D_BN_Leaky(512, 256, ksize=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=2))
        self.last2 = LastLayer(768, 256, numAnchor * (numClass + 5))
        self.up2 = nn.Sequential(DarknetConv2D_BN_Leaky(256, 128, ksize=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=2))
        self.last3 = LastLayer(384, 128, numAnchor * (numClass + 5))

    def forward(self, x):
        x = self.dark_conv1(x)  # 32x256x256
        x = self.res1(x)  # 64x128x128
        x = self.res2(x)  # 128x64x64
        x3 = self.res3(x)  # 256x32x32
        x4 = self.res4(x3)  # 512x16x16
        x5 = self.res5(x4)  # 1024x8x8

        x, y1 = self.last1(x5)  # 512x8x8,
        x = self.up1(x)  # 256x16x16
        x = torch.cat((x, x4), 1)  # 768x16x16
        x, y2 = self.last2(x)  # 256x16x16
        x = self.up2(x)  # 128x32x32
        x = torch.cat((x, x3), 1)  # 384x32x32
        x, y3 = self.last3(x)  # 128x32x32

        return y1, y2, y3

if __name__ == '__main__':
    net = Yolov3()
    X = torch.rand((16, 3, 416, 416))
    y1,y2,y3=net.forward(X)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)

    # for name, layer in net.named_children():
    #     X = layer(X)
    #     print(name, ' output shape:\t', X.shape)