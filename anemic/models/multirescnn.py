"""ICD Coding from Clinical Text Using Multi-Filter Residual
Convolutional Neural Network, 2020
https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network
"""  # 引用论文和代码库

from math import floor  # 向下取整用于卷积填充

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数接口
from torch.nn.init import xavier_uniform_ as xavier_uniform  # Xavier 参数初始化

from anemic.utils.mapper import ConfigMapper  # 名称到对象的映射器
from anemic.utils.model_utils import load_lookups  # 加载词表和标签映射
from anemic.utils.text_loggers import get_logger  # 日志工具

logger = get_logger(__name__)  # 初始化日志记录器


class WordRep(nn.Module):  # 词表示模块
    def __init__(self, config):
        super(WordRep, self).__init__()

        # 初始化词嵌入层
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        W = torch.Tensor(embedding_cls.load_emb_matrix(config.word2vec_dir))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()

        self.feature_size = self.embed.embedding_dim  # 词向量维度

        # 嵌入层 dropout
        self.embed_drop = nn.Dropout(p=config.dropout)

        # 不同卷积层的通道维度设置
        self.conv_dict = {
            1: [self.feature_size, config.num_filter_maps],
            2: [self.feature_size, 100, config.num_filter_maps],
            3: [self.feature_size, 150, 100, config.num_filter_maps],
            4: [self.feature_size, 200, 150, 100, config.num_filter_maps],
        }

    def forward(self, x):  # 前向传播
        x = self.embed(x)  # 查表得到词向量
        x = self.embed_drop(x)  # 应用 dropout
        return x  # 返回词向量序列


class OutputLayer(nn.Module):  # 输出层，包含标签注意力
    def __init__(self, Y, input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, Y)  # 注意力参数矩阵
        xavier_uniform(self.U.weight)  # 参数初始化

        self.final = nn.Linear(input_size, Y)  # 最终分类线性层
        xavier_uniform(self.final.weight)

    def forward(self, x):  # 计算标签注意力并输出
        self.alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        m = self.alpha.matmul(x)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return y


@ConfigMapper.map("models", "MultiCNN")  # 注册模型名称
class MultiCNN(nn.Module):
    def __init__(self, config):
        super(MultiCNN, self).__init__()

        Y = config.num_classes  # 标签数量
        self.dicts = load_lookups(
            dataset_dir=config.dataset_dir,
            mimic_dir=config.mimic_dir,
            static_dir=config.static_dir,
            word2vec_dir=config.word2vec_dir,
            version=config.version,
        )  # 加载词表等辅助数据

        self.word_rep = WordRep(config)  # 构建词表示模块

        if config.filter_size.find(",") == -1:  # 单一卷积核
            self.filter_num = 1
            filter_size = int(config.filter_size)
            self.conv = nn.Conv1d(
                self.word_rep.feature_size,
                config.num_filter_maps,
                kernel_size=filter_size,
                padding=floor(filter_size / 2),
            )
            xavier_uniform(self.conv.weight)
        else:  # 多卷积核并行
            self.filter_num = len(config.filter_size)
            self.conv = nn.ModuleList()
            for filter_size in config.filter_size:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(
                    self.word_rep.feature_size,
                    config.num_filter_maps,
                    kernel_size=filter_size,
                    padding=floor(filter_size / 2),
                )
                xavier_uniform(tmp.weight)
                self.conv.add_module("conv-{}".format(filter_size), tmp)

        self.output_layer = OutputLayer(
            Y, self.filter_num * config.num_filter_maps
        )  # 输出层

    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)  # 获取词向量
        x = x.transpose(1, 2)  # 调整维度以适配卷积

        if self.filter_num == 1:  # 单一卷积核
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:  # 多卷积核结果拼接
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, text_inputs)  # 计算输出
        return y, loss


class ResidualBlock(nn.Module):  # 残差卷积块
    def __init__(
        self, inchannel, outchannel, kernel_size, stride, use_res, dropout
    ):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv1d(
                inchannel,
                outchannel,
                kernel_size=kernel_size,
                stride=stride,
                padding=floor(kernel_size / 2),
                bias=False,
            ),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(
                outchannel,
                outchannel,
                kernel_size=kernel_size,
                stride=1,
                padding=floor(kernel_size / 2),
                bias=False,
            ),
            nn.BatchNorm1d(outchannel),
        )  # 两层卷积加批归一化

        self.use_res = use_res  # 是否使用残差
        if self.use_res:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    inchannel,
                    outchannel,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(outchannel),
            )  # 残差分支

        self.dropout = nn.Dropout(p=dropout)  # dropout

    def forward(self, x):  # 前向计算
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


@ConfigMapper.map("models", "ResCNN")  # 注册普通残差 CNN 模型
class ResCNN(nn.Module):
    def __init__(self, config):
        super(ResCNN, self).__init__()

        Y = config.num_classes  # 标签数量
        self.dicts = load_lookups(
            dataset_dir=config.dataset_dir,
            mimic_dir=config.mimic_dir,
            static_dir=config.static_dir,
            word2vec_dir=config.word2vec_dir,
            version=config.version,
        )  # 加载词表等

        self.word_rep = WordRep(config)  # 词表示模块

        self.conv = nn.ModuleList()  # 残差卷积堆叠
        conv_dimension = self.word_rep.conv_dict[config.conv_layer]
        for idx in range(config.conv_layer):
            tmp = ResidualBlock(
                conv_dimension[idx],
                conv_dimension[idx + 1],
                int(config.filter_size),
                1,
                True,
                config.dropout,
            )
            self.conv.add_module("conv-{}".format(idx), tmp)

        self.output_layer = OutputLayer(Y, config.num_filter_maps)

    def forward(self, x):
        x = self.word_rep(x)  # 得到词向量
        x = x.transpose(1, 2)
        for conv in self.conv:
            x = conv(x)
        x = x.transpose(1, 2)
        y, loss = self.output_layer(x)  # 分类并返回损失
        return y, loss


@ConfigMapper.map("models", "multirescnn")  # 注册 MultiResCNN 模型
class MultiResCNN(nn.Module):
    def __init__(self, config):
        super(MultiResCNN, self).__init__()

        Y = config.num_classes  # 标签数量
        self.dicts = load_lookups(
            dataset_dir=config.dataset_dir,
            mimic_dir=config.mimic_dir,
            static_dir=config.static_dir,
            word2vec_dir=config.word2vec_dir,
            version=config.version,
        )  # 读取辅助文件

        self.word_rep = WordRep(config)  # 词表示模块

        self.conv = nn.ModuleList()  # 多通道卷积

        self.filter_num = len(config.filter_size)  # 卷积核数量
        for filter_size in config.filter_size:
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(
                self.word_rep.feature_size,
                self.word_rep.feature_size,
                kernel_size=filter_size,
                padding=floor(filter_size / 2),
            )
            xavier_uniform(tmp.weight)
            one_channel.add_module("baseconv", tmp)

            conv_dimension = self.word_rep.conv_dict[config.conv_layer]
            for idx in range(config.conv_layer):
                tmp = ResidualBlock(
                    conv_dimension[idx],
                    conv_dimension[idx + 1],
                    filter_size,
                    1,
                    True,
                    config.dropout,
                )
                one_channel.add_module("resconv-{}".format(idx), tmp)
            self.conv.add_module("channel-{}".format(filter_size), one_channel)

        self.output_layer = OutputLayer(
            Y, self.filter_num * config.num_filter_maps
        )  # 输出层

    def forward(self, x):
        x = self.word_rep(x)  # 获取词向量
        x = x.transpose(1, 2)
        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        y = self.output_layer(x)  # 得到预测结果
        return y

    def get_input_attention(self):
        # 使用前向传播计算得到的注意力分数
        return self.output_layer.alpha.cpu().detach().numpy()
