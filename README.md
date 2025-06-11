# 中文对话生成模型 (Chinese Chatbot Model)

## 项目简介

这是一个基于Seq2Seq架构的中文对话生成模型项目，使用PyTorch实现。项目旨在构建一个能够理解中文输入并生成合理回复的对话系统。

## 主要功能

- 实现了标准的Encoder-Decoder架构
- 使用双向LSTM进行序列编码和解码
- 包含完整的数据预处理流程
- 支持中文分词和特殊字符处理

## 技术特点

1. **模型架构**:
   - Encoder: 双向LSTM编码器
   - Decoder: LSTM解码器
   - 注意力机制(待实现)

2. **数据处理**:
   - 中文分词(jieba)
   - 标点符号过滤
   - 特殊标记处理(PAD, UNK, END)
   - 序列长度标准化

3. **训练配置**:
   - 可调整的LSTM层数和隐藏层大小
   - Dropout防止过拟合
   - Adam优化器
   - 交叉熵损失函数

## 项目结构

```
.
├── data
    ├── conversations.corpus.json
├── config.py        # 配置文件(标点符号列表、数据路径等)
├── model.py         # 模型实现(Encoder, Decoder, Seq2Seq)
├── tools.py         # 数据处理工具和数据集类
└── train.py         # 训练脚本(待完善)
```

## 使用说明

1. 准备数据: 将对话数据放入`./data/conversations.corpus.json`
2. 配置参数: 在`config.py`中调整训练周期等参数
3. 运行训练: 执行`train.py`开始训练

## 后续计划

- <input type="checkbox" checked="checked"> 实现注意力机制
- <input type="checkbox"> 完善训练循环
- <input type="checkbox"> 添加模型评估指标

## 依赖项

- Python 3.8
- PyTorch
- jieba分词

欢迎贡献代码或提出改进建议！