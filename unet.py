import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop
class convLayers(nn.Module):
    def __init__(self, inp, out):
        super(convLayers, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
class unet(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        self.n_class = n_class
        #encoder
        self.conv1 = convLayers(3, 64)
        self.pool1= nn.MaxPool2d(2,2)
        self.conv2 = convLayers(64, 128)
        self.pool2= nn.MaxPool2d(2,2)
        self.conv3 = convLayers(128,256)
        self.pool3= nn.MaxPool2d(2,2)
        self.conv4 = convLayers(256, 512)
        self.pool4= nn.MaxPool2d(2,2)
        self.conv5 = convLayers(512, 1024)
        #decoder
        self.upConv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv6 = convLayers(1024,512)
        self.upConv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv7 = convLayers(512,256)
        self.upConv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv8 = convLayers(256,128)
        self.upConv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv9 = convLayers(128,64)
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)
    
    def forward(self,x):
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)
        x4 = self.conv4(p3)
        p4 = self.pool4(x4)
        x5 = self.conv5(p4)
        u1 = self.upConv1(x5)
        u1 = torch.cat((center_crop(x4,u1.shape[2:]),u1),1)
        y1 = self.conv6(u1)
        u2 = self.upConv2(y1)
        u2 = torch.cat((center_crop(x3,u2.shape[2:]),u2),1)
        y2 = self.conv7(u2)
        u3 = self.upConv3(y2)
        u3 = torch.cat((center_crop(x2,u3.shape[2:]),u3),1)
        y3 = self.conv8(u3)
        u4 = self.upConv4(y3)
        u4 = torch.cat((center_crop(x1,u4.shape[2:]),u4),1)
        y4 = self.conv9(u4)
        score = self.classifier(y4)
        return score    