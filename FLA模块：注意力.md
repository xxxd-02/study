#### FLA模块：注意力

```

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyAttentionalBlock(nn.Module):
    def __init__(self, plane, norm_layer=nn.BatchNorm2d):
        # 初始化函数，plane是输入和输出特征图的通道数，norm_layer是归一化层（默认为BatchNorm2d）
        super(FullyAttentionalBlock, self).__init__()
        # 定义两个全连接层，conv1和conv2
        self.conv1 = nn.Linear(plane, plane)
        self.conv2 = nn.Linear(plane, plane)
        
        # 定义卷积层 + 归一化层 + 激活函数（ReLU）
        self.conv = nn.Sequential(
            nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),  # 卷积操作
            norm_layer(plane),  # 归一化层
            nn.ReLU()  # ReLU激活函数
        )
        
        # 定义softmax操作，用于计算关系矩阵
        self.softmax = nn.Softmax(dim=-1)
        
        # 初始化可学习的参数gamma，用于调整最终的输出
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 前向传播过程，x为输入的特征图，形状为 (batch_size, channels, height, width)
        batch_size, _, height, width = x.size()
        
        # 对输入张量进行排列和变形，获取水平和垂直方向的特征    K（Key） 和 V（Value）
        feat_h = x.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)  # 水平方向特征         将维度重新变形为(batch_size * width, channels, height)，便于计算水平关系
        feat_w = x.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)  # 垂直方向特征         eat_w的维度调整为(batch_size * height, channels, width)，用于计算垂直关系。
        
        # 对输入张量分别在水平方向和垂直方向进行池化，并通过全连接层进行编码      Q（Query）
        encode_h = self.conv1(F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())  # 水平方向编码
        encode_w = self.conv2(F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())  # 垂直方向编码
        
        # 计算水平方向和垂直方向的关系矩阵      A
        energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))  # 计算水平方向的关系
        energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))  # 计算垂直方向的关系
        
        # 计算经过softmax后的关系矩阵
        full_relation_h = self.softmax(energy_h)  # 水平方向的关系
        full_relation_w = self.softmax(energy_w)  # 垂直方向的关系
        
        # 通过矩阵乘法和关系矩阵，对特征进行加权和增强   A V 融合
        full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3, 1)  # 水平方向的增强
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)  # 垂直方向的增强
        
        # 将水平和垂直方向的增强特征进行融合，并加上原始输入特征
        out = self.gamma * (full_aug_h + full_aug_w) + x
        
        # 通过卷积层进行进一步的特征处理
        out = self.conv(out)
        
        return out  # 返回处理后的特征图
```

