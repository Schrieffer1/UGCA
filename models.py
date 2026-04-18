import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer


class residual_block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

    def forward(self, x):  # (1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True)  # (1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2)  # (1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1 + x3, inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        return out


class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, out_channel3=32):

        super(D_Res_3d_CNN, self).__init__()

        # --- 第1层提取 ---
        self.block1 = residual_block(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))

        # --- 第2层提取 ---
        self.block2 = residual_block(out_channel1, out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))

        # --- 第3层提取 (不带池化以保持空间尺寸) ---
        # 输入为第2层输出(out_channel2)，输出为新定义的 out_channel3
        self.block3 = residual_block(out_channel2, out_channel3)

        # --- 第4层提取 (原 Final Conv 层) ---
        self.conv = nn.Conv3d(in_channels=out_channel3, out_channels=32, kernel_size=3, bias=False)

    def forward(self, x):
        # 输入形状: (Batch, 100, 9, 9)
        x = x.unsqueeze(1)  # -> (Batch, 1, 100, 9, 9)

        # --- Layer 1 ---
        x = self.block1(x)
        x = self.maxpool1(x)  # 输出形状约: (Batch, 8, 25, 5, 5)

        # --- Layer 2 ---
        x = self.block2(x)
        x = self.maxpool2(x)  # 输出形状约: (Batch, 16, 7, 3, 3)

        # --- Layer 3 ---
        # 经过残差块处理，特征语义加深，但维度保持不变 (Padding=1, Stride=1)
        # 输出形状: (Batch, 32, 7, 3, 3) 
        x = self.block3(x)

        # --- Layer 4 (Final) ---
        # 3x3x3 卷积，无Padding
        # 光谱维: 7 - 3 + 1 = 5
        # 空间维: 3 - 3 + 1 = 1
        # 输出形状: (Batch, 32, 5, 1, 1)
        x = self.conv(x)

        # 展平
        x = x.view(x.shape, -1)  # -> (Batch, 160)
        return x

#############################################################################################################

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class DomainClassifier(nn.Module):
    def __init__(self):# torch.Size([1, 64, 7, 3, 3])
        super(DomainClassifier, self).__init__() #
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024), #nn.Linear(320, 512), nn.Linear(FEATURE_DIM*CLASS_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.domain = nn.Linear(1024, 1) # 512

    def forward(self, x, iter_num):
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10,10000.0)
        x.register_hook(grl_hook(coeff))
        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        # print("映射后的域特征f",return_tensor.shape)
        # print("映射后的概率向量g",return_list[1].shape)
        # for single in return_list[1:]:
        #     return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Center Loss 用于增强特征空间的类内紧凑性。
    参考论文: Wen et al. "A Discriminative Feature Learning Approach for Deep Face Recognition"
    """

    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        # 初始化类中心参数，随机初始化
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: 特征矩阵，形状为 (batch_size, feat_dim)
            labels: 真实标签，形状为 (batch_size)
        """
        batch_size = x.size(0)

        # 计算每个特征与所有类中心的平方欧氏距离
        # distmat = x^2 + c^2 - 2xc
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        # 获取每个样本对应真实类别的距离
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()

        # 将labels扩展为矩阵形式以便应用掩码
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        # 限制最小值防止数值不稳定，计算平均损失
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
