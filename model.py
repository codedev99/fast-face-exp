import torch
from torch import nn

# EmotionNetBlock
class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super(InvertedBottleneck, self).__init__()
        self.leaky = nn.LeakyReLU(0.2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(in_channels))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))
    
    def forward(self, x):
        m = self.conv2(self.leaky(self.conv1(x)))
        m = x + m
        m = self.leaky(self.conv3(self.leaky(m)))

        return m


# EmotionNet
class EmotionNet(nn.Module):
    def __init__(self, num_emotions):
        super(EmotionNet, self).__init__()
        self.num_emotions = num_emotions
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 24, 3, 2, bias=False), # output_size = 24x24
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2))
        self.layer2 = InvertedBottleneck(24, 40, 3, 4) # output_size = 12x12
        self.layer3 = InvertedBottleneck(40, 56, 3, 4) # output_size = 6x6
        self.layer4 = InvertedBottleneck(56, 72, 3, 4) # output_size = 6x6
        self.avgpool = nn.AvgPool2d(6, 1, 0) # output_size = 1x1
        self.layer5 = nn.Linear(72, self.num_emotions)
        self.pool = nn.MaxPool2d(3,2,1)
    
    def forward(self, x):
        x = self.avgpool(self.layer4(self.pool(self.layer3(self.pool(self.layer2(self.layer1(x)))))))
        x = x.reshape(-1, 72)
        x = self.layer5(x)
        
        return x