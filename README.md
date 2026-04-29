# 机器学习项目说明

## 项目结构

```bash
ML_Project/
├── data/
│   ├── titanic/              # Titanic 数据集(train.csv、test.csv)
│   ├── mnist/                # MNIST 图像数据
│   ├── cifar10_images/       # CIFAR-10 图像数据(train/、test/)
│   └── house_data.csv        # 房价回归数据
├── src/
│   ├── data_preprocess.py    # 数据预处理与特征提取
│   ├── ANN.py                # ANN 回归/二分类/多分类实现
│   ├── KNN.py
│   ├── logistic_regression.py
│   ├── linear_regression.py
│   └── SVM.py
├── models/                   # 保存训练后的模型
├── results/                  # 保存准确率与 loss 曲线图
├── main.py                   # 通用入口
├── requirements.txt
└── README.md
```

ANN 的核心算法写在 `src/ANN.py` 中，统一通过 `main.py` 调用。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 已实现算法

1. `KNN`：标准 K 近邻分类
2. `Logistic Regression`：二分类逻辑回归
3. `Linear Regression`：线性回归
4. `SVM`：支持向量机二分类
5. `ANN`：人工神经网络，支持房价回归、Titanic 二分类、CIFAR-10 多分类

## 数据集说明

- `house`：回归任务，预测房价
- `titanic`：二分类任务，预测乘客是否存活
- `mnist`：手写数字图像分类
- `cifar10`：10 类图像分类

## 使用方法

### 1. 通用入口 `main.py`

适用于 `knn`、`logistic`、`linear`、`svm` 和 `ann`。

```bash
# 线性回归训练房价数据
python main.py --algo=linear --data=house --process=train

# 线性回归测试房价数据
python main.py --algo=linear --data=house --process=test

# ANN 训练房价回归
python main.py --algo=ann --data=house --process=train --epochs=300 --learning_rate=0.001 --batch_size=32

# ANN 测试房价回归
python main.py --algo=ann --data=house --process=test

# ANN 训练 Titanic
python main.py --algo=ann --data=titanic --process=train

# ANN 测试 Titanic
python main.py --algo=ann --data=titanic --process=test

# ANN 训练 CIFAR10
python main.py --algo=ann --data=cifar10 --process=train

# ANN 测试 CIFAR10
python main.py --algo=ann --data=cifar10 --process=test

# KNN 训练 Titanic
python main.py --algo=knn --data=titanic --process=train

# Logistic Regression 训练 MNIST（二分类）
python main.py --algo=logistic --data=mnist --process=train

# SVM 训练 CIFAR10（二分类）
python main.py --algo=svm --data=cifar10 --process=train --class1=2 --class2=3
```

### 2. ANN 任务说明

目前 ANN 支持以下任务，全部通过 `main.py` 统一调用：

- `house`：房价回归
- `titanic`：二分类
- `cifar10`：多分类

其中：

- Titanic ANN 会读取 `data/titanic/train.csv` 和 `data/titanic/test.csv`
- CIFAR10 ANN 会读取 `data/cifar10_images/train/` 和 `data/cifar10_images/test/`
- CIFAR10 为提高训练效率，会先提取 HOG 特征，再训练多分类 ANN
- 准确率写入 `results/accuracy.txt`
- loss 曲线图保存到 `results/` 目录

## 输出结果

训练或测试完成后，结果会保存在以下位置：

- `models/`：保存 `.pkl` 模型文件
- `results/accuracy.txt`：保存每次测试或验证的准确率记录
- `results/*_loss.jpg`：保存 loss 曲线图

## 当前已验证功能

- 房价数据 `house_data.csv` 的 ANN 回归已完成训练和测试
- Titanic 数据 ANN 二分类已完成训练和测试
- CIFAR-10 数据 ANN 多分类已可通过 `main.py` 调用
