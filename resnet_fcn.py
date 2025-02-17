import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights

class ResNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.n_class = n_class
        self.resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # Here we need to get rid of the last two layers
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-2])
        
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    
    def forward(self, x):
        # Complete the forward function for the rest of the encoder
        x1 = self.encoder(x)

        # Complete the forward function for the rest of the decoder
        y1 = self.bn1(self.relu(self.deconv1(x1)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))


        score = self.classifier(y5)

        return score  # size=(N, n_class, H, W)
