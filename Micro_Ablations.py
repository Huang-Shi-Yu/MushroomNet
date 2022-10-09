import torch
import torch.nn as nn
from torchsummary import summary


# Micro
class MushroomNet_Micro(nn.Module):
    def __init__(self):
        super(MushroomNet_Micro,self).__init__()

        #卷积和池化层
        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 32, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=16, out_features=9)
        )

        # 定义池化层，规定输出size
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)
        return x



# Larger FC
class MushroomNet_Micro_Larger_FC(nn.Module):
    def __init__(self):
        super(MushroomNet_Micro_Larger_FC,self).__init__()

        #卷积和池化层
        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 32, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=9)
        )

        # 定义池化层，规定输出size
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)
        return x


# More channels
class MushroomNet_Micro_More_Channels(nn.Module):
    def __init__(self):
        super(MushroomNet_Micro_More_Channels,self).__init__()

        #卷积和池化层
        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 64, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=16, out_features=9)
        )

        # 定义池化层，规定输出size
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)
        return x


# Clip 1
class MushroomNet_Micro_Clip1(nn.Module):
    def __init__(self):
        super(MushroomNet_Micro_Clip1,self).__init__()

        #卷积和池化层
        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, 3, 4, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 32, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=16, out_features=9)
        )

        # 定义池化层，规定输出size
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)
        return x



# Clip 1
class MushroomNet_Micro_Clip2(nn.Module):
    def __init__(self):
        super(MushroomNet_Micro_Clip2,self).__init__()

        #卷积和池化层
        self.conv=nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8, padding=0),
        )

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 3, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=16, out_features=9)
        )

        # 定义池化层，规定输出size
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)
        return x

