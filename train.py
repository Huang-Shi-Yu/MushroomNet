import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import argparse
import time

import sys
# sys.path.append("MN_Models")
import MN_Models as MN
import Micro_Ablations as MA
import Micro_Functions as MF

# 设置cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 选择模型
model_option = sys.argv[1]


#基础比较模型
if model_option == "Micro":
    model = MN.MushroomNet_Micro()
    model = model.to(device=device)

elif model_option == "Slim":
    model = MN.MushroomNet_Slim()
    model = model.to(device=device)
elif model_option == "Middle":
    model = MN.MushroomNet_Middle()
    model = model.to(device=device)
elif model_option == "Large":
    model = MN.MushroomNet_Large()
    model = model.to(device=device)
elif model_option == "VGG11":
    model = torchvision.models.vgg11(pretrained=False, num_classes=9)
    model = model.to(device=device)
elif model_option == "ResNet18":
    model = torchvision.models.resnet18(pretrained=False, num_classes=9)
    model = model.to(device=device)
elif model_option == "ResNet34":
    model = torchvision.models.resnet34(pretrained=False, num_classes=9)
    model = model.to(device=device)
elif model_option == "MobileNetV3":
    model = torchvision.models.mobilenet_v3_small(pretrained=False, num_classes=9)
    model = model.to(device=device)
elif model_option == "ShuffleNetV2":
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=False, num_classes=9)
    model = model.to(device=device)

#消融比较模型
elif model_option == "Larger-FC":
    model = MA.MushroomNet_Micro_Larger_FC()
    model = model.to(device=device)
elif model_option == "More-channels":
    model = MA.MushroomNet_Micro_More_Channels()
    model = model.to(device=device)
elif model_option == "Clip1":
    model = MA.MushroomNet_Micro_Clip1()
    model = model.to(device=device)
elif model_option == "Clip2":
    model = MA.MushroomNet_Micro_Clip2()
    model = model.to(device=device)

# 激活函数比较模型
elif model_option == "Leaky_ReLU":
    model = MF.MushroomNet_Micro_Larger_FC_Leaky_ReLU()
    model = model.to(device=device)
elif model_option == "ELU":
    model = MF.MushroomNet_Micro_Larger_FC_ELU()
    model = model.to(device=device)

# 数据集比较
elif model_option == "MicroV2-half-LQ":
    model = MA.MushroomNet_Micro_Larger_FC()
    model = model.to(device=device)

elif model_option == "MicroV2-MQ":
    model = MA.MushroomNet_Micro_Larger_FC()
    model = model.to(device=device)

elif model_option == "MicroV2-HQ":
    model = MA.MushroomNet_Micro_Larger_FC()
    model = model.to(device=device)


data_root = " "
# 根据模型选择数据集
if model_option in ["Micro","Slim","Larger-FC","More-channels","Clip1","Clip2","Leaky_ReLU","ELU"]:
    data_root = "./Datasets/LQ"
elif model_option in ["Middle","Large","MicroV2-MQ"]:
    data_root = "./Datasets/MQ"
elif model_option in ["MicroV2-half-LQ"]:
    data_root = "./Datasets/half_LQ"
else:
    data_root = "./Datasets/HQ"
print(data_root)
dataset = torchvision.datasets.ImageFolder(root=data_root,transform=torchvision.transforms.ToTensor())


# 划分训练集和验证集
train_size = int(len(dataset)*0.8)
test_size = len(dataset) - train_size
train_dataset,val_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)

# 设置损失函数，使用交叉熵损失
cross_entropy_loss = torch.nn.CrossEntropyLoss()
cross_entropy_loss = cross_entropy_loss.to(device=device)

# 设置学习率和优化器，使用L2正则化，正则化参数为0.0001
learning_rate = 1e-3
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

# 设置epoch
epoch = 30
total_train_step = 0

# 统计运行时间
start = time.perf_counter()


for i in range(epoch):
    print(" -----------------the {} number of training epoch --------------".format(i + 1))
    epoch_train_step = 0
    epoch_val_step = 0
    model.train()

    # 初始化参数列表
    loss_train = 0
    loss_val = 0
    train_true_total = 0
    val_true_total = 0
    acc_train = 0
    acc_val = 0

    # 开始训练
    for data in train_loader:
        # 图片数据和标签
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 计算损失函数，使用交叉熵损失
        outputs = model(imgs)
        loss_train = cross_entropy_loss(outputs, targets)

        # 反向传播
        optim.zero_grad()
        loss_train.backward()
        optim.step()

        # 计数器
        total_train_step = total_train_step + 1
        epoch_train_step = epoch_train_step + 1
        #print(epoch_train_step)

        # 计算模型精度
        predicted = torch.max(outputs.data,1)[1]
        train_true_total += (predicted == targets).sum()
        acc_train = int(train_true_total)/(epoch_train_step * 64)

    # 显示训练集损失
    print("the training epoch is {} and its train loss of model is {}".format(i+1, loss_train.item()))
    # 显示训练集精度
    print("the training epoch is {} and its train accuracy of model is {}".format(i+1, acc_train))

    # 开始验证
    for data in val_loader:
        # 图片数据和标签
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 计算损失函数，使用交叉熵损失
        outputs = model(imgs)
        loss_val = cross_entropy_loss(outputs, targets)

        # 计数器
        epoch_val_step = epoch_val_step + 1

        # 计算模型精度
        predicted = torch.max(outputs.data, 1)[1]
        # print(predicted)
        val_true_total += (predicted == targets).sum()
        # print(int(true_total))
        acc_val = int(val_true_total) / (epoch_val_step * 64)


    # epoch结束，显示结果
    # 显示验证集损失
    print("the training epoch is {} and its val loss of model is {}".format(i+1, loss_val.item()))
    # 显示训练集精度
    print("the training epoch is {} and its val accuracy of model is {}".format(i+1, acc_val))
    #writer.add_scalar("train_loss", loss_train.item(), total_train_step)
    if i == (epoch - 1):
        torch.save(model.state_dict(), "Model_save/"+model_option+".pth")
        print("the model of last training step was saved! ")

# 输出程序运行时间
end = time.perf_counter()
print("time:",end-start)



