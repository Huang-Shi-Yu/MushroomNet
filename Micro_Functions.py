import torch
import torch.nn as nn
from torchsummary import summary


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


# Larger FC + LeakyReLU
class MushroomNet_Micro_Larger_FC_Leaky_ReLU(nn.Module):
    def __init__(self):
        super(MushroomNet_Micro_Larger_FC_Leaky_ReLU,self).__init__()

        #卷积和池化层
        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 2, 0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 32, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
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



# Larger FC + ELU
class MushroomNet_Micro_Larger_FC_ELU(nn.Module):
    def __init__(self):
        super(MushroomNet_Micro_Larger_FC_ELU,self).__init__()

        #卷积和池化层
        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 0),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 32, out_features=256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ELU(),
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






