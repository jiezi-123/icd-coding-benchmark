# 导入所需的库和工具
import argparse  # 解析命令行参数
import os  # 操作系统相关功能（未在本脚本中使用）

import pandas  # 用于数据处理
import torch  # PyTorch 深度学习框架
from torchsummaryX import summary  # 打印模型结构

from anemic.utils.configuration import Config  # 读取 YAML 配置
from anemic.utils.import_related_ops import (  # 设置 pandas 显示
    pandas_related_ops,
)
from anemic.utils.mapper import ConfigMapper  # 名称到对象的映射工具
from anemic.utils.misc import seed  # 随机种子设置

pandas_related_ops()  # 调整 pandas 的显示选项


# 解析命令行参数
parser = argparse.ArgumentParser(description="Train or test the model")
parser.add_argument(
    "--config_path", type=str, action="store", help="Path to the config file"
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Whether to use validation data or test data",
    default=False,
)
parser.add_argument(
    "--model_summary",
    action="store_true",
    help="Whether to print model summary. Note that this is supported only for "
    "models which take in a 2D input. This will be extended later",
    default=False,
)
args = parser.parse_args()  # 解析参数

# 加载配置文件
config = Config(path=args.config_path)

if not args.test:  # 进入训练模式
    # 设置随机种子保证可复现
    seed(config.trainer.params.seed)

    # 构建训练和验证数据集
    train_data = ConfigMapper.get_object("datasets", config.dataset.name)(
        config.dataset.params.train
    )
    val_data = ConfigMapper.get_object("datasets", config.dataset.name)(
        config.dataset.params.val
    )

    # 根据配置初始化模型
    model = ConfigMapper.get_object("models", config.model.name)(
        config.model.params
    )

    if args.model_summary:  # 打印模型结构信息
        summary(
            model, torch.randint(low=0, high=50000, size=(1, 20)).to(torch.long)
        )

    # 创建训练器
    trainer = ConfigMapper.get_object("trainers", config.trainer.name)(
        config.trainer.params
    )

    # 开始训练
    trainer.train(model, train_data, val_data)
else:  # 进入测试模式
    # 仅构建测试集
    test_data = ConfigMapper.get_object("datasets", config.dataset.name)(
        config.dataset.params.test
    )

    # 初始化模型
    model = ConfigMapper.get_object("models", config.model.name)(
        config.model.params
    )

    # 创建训练器（用于加载和评估）
    trainer = ConfigMapper.get_object("trainers", config.trainer.name)(
        config.trainer.params
    )

    # 在测试集上评估模型
    trainer.test(model, test_data)
