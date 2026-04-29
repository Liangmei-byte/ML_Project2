import os
import numpy as np
import cv2
from skimage.feature import hog
import pandas as pd

def load_titanic_data():
    train_path = os.path.join('data', 'titanic', 'train.csv')
    test_path = os.path.join('data', 'titanic', 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def preprocess_titanic_data(df):
    df = df.copy()
    
    # 标准化列名
    df.columns = df.columns.str.lower()
    
    # 处理列名映射
    if '2urvived' in df.columns:
        df.rename(columns={'2urvived': 'survived'}, inplace=True)
    
    # 只保留需要的列
    required_columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']
    df = df[[col for col in required_columns if col in df.columns]]
    
    # 填充缺失值
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    if 'embarked' in df.columns:
        df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    
    # 处理性别
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({0: 0, 1: 1})
    
    # 处理 embark
    if 'embarked' in df.columns:
        df = pd.get_dummies(df, columns=['embarked'], drop_first=True)
    
    # 构建特征列表
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
    if 'embarked_q' in df.columns:
        features.append('embarked_q')
    if 'embarked_s' in df.columns:
        features.append('embarked_s')
    
    X = df[features].values
    
    if 'survived' in df.columns:
        y = df['survived'].values
        return X, y
    else:
        return X

def load_house_data():
    data_path = os.path.join('data', 'house_data.csv')
    df = pd.read_csv(data_path)
    
    # 将所有列转换为数值类型，无效值转换为NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 填充NaN值
    df = df.fillna(df.median())
    
    X = df.drop('y', axis=1).values
    y = df['y'].values
    
    return X, y

def load_mnist_data():
    data_dir = os.path.join('data', 'mnist')
    X = []
    y = []
    
    for label in range(10):
        label_dir = os.path.join(data_dir, str(label))
        if not os.path.exists(label_dir):
            continue
        
        for img_file in os.listdir(label_dir):
            if img_file.endswith('.png'):
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    X.append(img.flatten())
                    y.append(label)
    
    return np.array(X), np.array(y)

def load_cifar10_data():
    data_dir = os.path.join('data', 'cifar10_images')
    X = []
    y = []
    
    # 类别名称到数字标签的映射
    class_map = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }
    
    # 加载train和test目录
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        for class_name, label in class_map.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_file in os.listdir(class_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        X.append(img.flatten())
                        y.append(label)
    
    return np.array(X), np.array(y)


def load_cifar10_split_data():
    data_dir = os.path.join('data', 'cifar10_images')
    class_map = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }

    def _load_split(split):
        X = []
        y = []
        split_dir = os.path.join(data_dir, split)
        for class_name, label in class_map.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_file in os.listdir(class_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        X.append(img.flatten())
                        y.append(label)

        return np.array(X), np.array(y)

    X_train, y_train = _load_split('train')
    X_test, y_test = _load_split('test')
    return X_train, y_train, X_test, y_test

def extract_hog_features(images):
    hog_features = []
    for img in images:
        if len(img.shape) == 1:
            if img.shape[0] == 784:  # MNIST
                img = img.reshape(28, 28)
            elif img.shape[0] == 3072:  # CIFAR10
                img = img.reshape(32, 32, 3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(features)
    
    return np.array(hog_features)

def get_binary_class_data(X, y, class1, class2):
    mask = (y == class1) | (y == class2)
    X_binary = X[mask]
    y_binary = y[mask]
    y_binary = np.where(y_binary == class1, 0, 1)
    
    return X_binary, y_binary