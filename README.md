# CutPaste-paddle

## 目录

- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 目录介绍]()
    - [3.2 准备环境]()
    - [3.3 准备数据]()
    - [3.4 准备模型]()
- [4. 开始使用]()
    - [4.1 模型训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
- [5. 模型推理部署]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 参考链接与文献]()

## 1. 简介
cutpaste是一种简单有效的自监督学习方法，其目标是构建一个高性能的两阶段缺陷检测模型，在没有异常数据的情况下检测图像的未知异常模式。首先通过cutpaste数据增强方法学习自监督深度表示，然后在学习的表示上构建生成的单类分类器，从而实现自监督的异常检测。

**论文:** [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/pdf/2104.04015v1.pdf)

**参考repo:** [pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste)

在此非常感谢`Runinho`等人贡献的[pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste) ，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

- 数据集大小：共包含15个物品类别，解压后总大小在4.92G左右
- 数据集下载链接：[mvtec](https://pan.baidu.com/s/1KQR4kIBHJVDlnaO1SpqyKg?pwd=0722) 提取码：0722  解压到images文件夹
- 训练权重下载链接：[pdparams & pkl](https://pan.baidu.com/s/1stWIojfxf69c630R9q62xA?pwd=0722) 提取码：0722  解压到models_param文件夹
- 
# 复现精度（Comparison to Li et al.）
| defect_type   | Runinho. CutPaste (3-way) | Li et al. CutPaste (3-way) | CutPaste (3-way) 复现|
|:--------------|-------------------:|-----------------------------:|-----------------------:|
| bottle        |               99.6 |                         98.3 |                   100.0 |
| cable         |               77.2 |                         80.6 |                    92.4 |
| capsule       |               92.4 |                         96.2 |                    92.9 |
| carpet        |               60.1 |                         93.1 |                    89.8 |
| grid          |              100.0 |                         99.9 |                    99.9 |
| hazelnut      |               86.8 |                         97.3 |                    96.6 |
| leather       |              100.0 |                        100.0 |                   100.0 |
| metal_nut     |               87.8 |                         99.3 |                    96.0 |
| pill          |               91.7 |                         92.4 |                    95.5 |
| screw         |               86.8 |                         86.3 |                    86.3 |
| tile          |               97.2 |                         93.4 |                    99.0 |
| toothbrush    |               94.7 |                         98.3 |                    99.2 |
| transistor    |               93.0 |                         95.5 |                    95.6 |
| wood          |               99.4 |                         98.6 |                    98.8 |
| zipper        |               98.8 |                         99.4 |                   100.0 |
| average       |               91.0 |                         95.2 |                    96.1 |


## 3. 准备数据与环境

### 3.1 目录介绍

```
paddle_autec
    |--demo                            # 测试使用的样例图片，两张
    |--deploy                          # 预测部署相关
        |--export_model.py             # 导出模型
        |--infer.py                    # 部署预测
    |--images                          # 训练和测试数据集
    |--lite_data                       # 自建立的小数据集，只有bottle和grid两个
    |--logs                            # 训练train和测试eval打印的日志信息  
    |--models_param                    # 训练的模型权
    |--test_tipc                       # tipc代码
    |--tools                           # 工具类文件
        |--cutpaste.py                 # 论文代码
        |--dataset.py                  # 数据加载
        |--density.py                  # 高斯聚类代码
        |--eval.py                     # 评估代码
        |--model.py                    # 论文模型 res18
        |--predict.py                  # 预测代码
        |--train.py                    # 训练代码
    |----README.md                     # 用户手册
    |----requirements.txt              # 依赖包
```

### 3.2 准备环境

首先介绍下支持的硬件和框架版本等环境的要求：

- 硬件：GPU显存建议在6G以上
- 框架：
  - PaddlePaddle >= 2.3.1
- 环境配置：直接使用`pip install -r requirements.txt`安装依赖即可。

### 3.3 准备数据

- 全量数据训练：
  - 下载好 [metec](https://pan.baidu.com/s/1KQR4kIBHJVDlnaO1SpqyKg?pwd=0722) 提取码：0722 数据集
  - 将其解压到 **images** 文件夹下
- 少量数据训练：
  - 无需下载数据集，使用lite_data里的数据即可


### 3.4 准备模型

- 默认使用resnet18预训练模型进行训练，如想关闭,需要传入参数：`python train.py --no_pretrained`

## 4. 开始使用


### 4.1 模型训练

- 全量数据训练：
  - 下载好 [metec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad/) 数据集后，将其解压到 **./images** 文件夹下
  - 运行指令`python tools/train.py --epochs 256 --batch_size 96 --cuda True`
- 少量数据训练：
  - 运行指令`python tools/train.py --data_dir lite_data --type bottle --epochs 5 --batch_size 4 --cuda False`
- 部分训练日志如下所示：
```
############################
training bottle
############################

Type : bottle Train [ Epoch 1/256 iter 1 ] lr: 0.03000, loss: 1.34867, avg_reader_cost: 2.85896 sec, avg_batch_cost: 5.08064 sec, avg_samples: 96.0, avg_ips: 18.89526 images/sec.
Type : bottle Train [ Epoch 1/256 iter 2 ] lr: 0.03000, loss: 0.71292, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.10269 sec, avg_samples: 96.0, avg_ips: 934.85698 images/sec.
Type : bottle Train [ Epoch 1/256 iter 3 ] lr: 0.03000, loss: 0.65832, avg_reader_cost: 0.00009 sec, avg_batch_cost: 0.03219 sec, avg_samples: 17.0, avg_ips: 528.06209 images/sec.
Type : bottle Train [ Epoch 2/256 iter 1 ] lr: 0.03000, loss: 0.55255, avg_reader_cost: 2.56750 sec, avg_batch_cost: 2.67233 sec, avg_samples: 96.0, avg_ips: 35.92374 images/sec.
Type : bottle Train [ Epoch 2/256 iter 2 ] lr: 0.03000, loss: 0.36559, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.10577 sec, avg_samples: 96.0, avg_ips: 907.63495 images/sec.
Type : bottle Train [ Epoch 2/256 iter 3 ] lr: 0.03000, loss: 0.37505, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.03266 sec, avg_samples: 17.0, avg_ips: 520.54467 images/sec.
Type : bottle Train [ Epoch 3/256 iter 1 ] lr: 0.02998, loss: 0.37115, avg_reader_cost: 1.98013 sec, avg_batch_cost: 2.08481 sec, avg_samples: 96.0, avg_ips: 46.04729 images/sec.
Type : bottle Train [ Epoch 3/256 iter 2 ] lr: 0.02998, loss: 0.25799, avg_reader_cost: 0.14576 sec, avg_batch_cost: 0.24890 sec, avg_samples: 96.0, avg_ips: 385.70197 images/sec.
...
``` 


### 4.2 模型评估

- 全量数据模型评估：`python eval.py --cuda True`
- 少量数据模型评估：`python tools/eval.py --data_dir lite_data --type bottle --cuda False`
```
Namespace(cuda=True, density='paddle', head_layer=1, model_dir='../models_param2/mvtec', save_plots=True, type='all')
evaluating bottle
loading model models_param2/mvtec/model_bottle
[512, 128]
bottle AUC: 1.0
evaluating cable
loading model models_param2/mvtec/model_cable
[512, 128]
cable AUC: 0.9246626686656673
evaluating capsule
loading model models_param2/mvtec/model_capsule
[512, 128]
capsule AUC: 0.9285999202233746
evaluating carpet
loading model models_param2/mvtec/model_carpet
[512, 128]
carpet AUC: 0.8976725521669342
evaluating grid
loading model models_param2/mvtec/model_grid
[512, 128]
grid AUC: 0.9991645781119465
``` 
预测分数：12.288496017456055 < 最佳阈值： 53.726806640625 
预测结果：正常数据
### 4.3 模型预测（需要预先完成4.1训练 或 直接将训练权值下载后解压到models_param文件夹中）

- 基于原始代码的模型预测：`python tools/predict.py --data_type bottle --img_file demo/bottle_good.png`

部分结果如下：
```
预测分数：12.288496017456055 < 最佳阈值： 53.726806640625 
预测结果：正常数据
```

- 基于推理引擎的模型预测：
```
python deploy/export_model.py
python deploy/infer.py --defect_type bottle --img_path demo/bottle_good.png
```
部分结果如下：
```
> python deploy/export_model.py
inference model has been saved into deploy

> python deploy/infer.py --data_type bottle --img_path demo/bottle_good.png 
image_name: demo/bottle_good.png, data is normal, score is 12.283802032470703, threshold is 53.726806640625
``` 


## 5. 模型推理部署

模型推理部署详见4.3节-基于推理引擎的模型预测。


## 6. 自动化测试脚本

[tipc创建及基本使用方法。](https://github.com/PaddlePaddle/models/blob/release/2.2/tutorials/tipc/train_infer_python/test_train_infer_python.md)


## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献
**参考论文:** [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/pdf/2104.04015v1.pdf)

**参考repo:** [pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste)
