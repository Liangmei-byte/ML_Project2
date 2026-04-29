import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data_preprocess import (
    load_titanic_data, preprocess_titanic_data, load_house_data,
    load_mnist_data, load_cifar10_data, extract_hog_features, get_binary_class_data
)
from src.KNN import knn_train, knn_test
from src.logistic_regression import logistic_regression_train, logistic_regression_test
from src.linear_regression import linear_regression_train, linear_regression_test
from src.SVM import svm_train, svm_test
from src.ANN import (
    standardize_fit_transform, standardize_transform,
    ann_train_regression, ann_test_regression,
    ann_train_titanic, ann_test_titanic,
    ann_train_cifar10, ann_test_cifar10
)

def save_model(model, algo, data, date):
    model_path = os.path.join('models', f'{algo}_{data}_{date}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def load_model(algo, data, date):
    model_path = os.path.join('models', f'{algo}_{data}_{date}.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_result(result, algo, data, date, result_type):
    if result_type == 'loss':
        result_path = os.path.join('results', f'{algo}_{data}_{date}_loss.jpg')
        plt.figure(figsize=(10, 6))
        if isinstance(result, dict):
            if 'train' in result:
                plt.plot(result['train'], label='Train Loss')
            if 'val' in result:
                plt.plot(result['val'], label='Validation Loss')
            if 'train' in result or 'val' in result:
                plt.legend()
        elif result:
            plt.plot(result)
        else:
            # 对于不需要迭代的算法（如最小二乘法），显示一条水平直线
            plt.axhline(y=0, color='r', linestyle='-')
            plt.text(0.5, 0.5, 'No loss curve for non-iterative algorithm', 
                     ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{algo} {data} Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(result_path)
        plt.close()
        print(f"Loss curve saved to {result_path}")
    elif result_type == 'accuracy':
        result_path = os.path.join('results', 'accuracy.txt')
        with open(result_path, 'a') as f:
            f.write(f'{datetime.now()}: {algo} on {data} - Accuracy: {result:.4f}\n')
        print(f"Accuracy saved to {result_path}")

def main():
    parser = argparse.ArgumentParser(description='Machine Learning Project')
    parser.add_argument('--algo', type=str, required=True, choices=['knn', 'logistic', 'linear', 'svm', 'ann'])
    parser.add_argument('--data', type=str, required=True, choices=['titanic', 'house', 'mnist', 'cifar10'])
    parser.add_argument('--process', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--k', type=int, default=3, help='K for KNN')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for gradient descent')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--class1', type=int, default=0, help='First class for binary classification')
    parser.add_argument('--class2', type=int, default=1, help='Second class for binary classification')
    
    args = parser.parse_args()
    date = datetime.now().strftime('%Y%m%d')
    
    if args.data == 'titanic':
        train_df, test_df = load_titanic_data()
        X_train, y_train = preprocess_titanic_data(train_df)
        # 预处理测试数据
        result = preprocess_titanic_data(test_df)
        if isinstance(result, tuple):
            X_test, y_test = result
        else:
            X_test = result
            y_test = None
    
    elif args.data == 'house':
        X, y = load_house_data()
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    
    elif args.data == 'mnist':
        X, y = load_mnist_data()
        if args.algo in ['logistic', 'svm']:
            X = extract_hog_features(X)
            X, y = get_binary_class_data(X, y, args.class1, args.class2)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    
    elif args.data == 'cifar10':
        X, y = load_cifar10_data()
        if args.algo in ['logistic', 'svm']:
            X = extract_hog_features(X)
            X, y = get_binary_class_data(X, y, args.class1, args.class2)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    
    if args.process == 'train':
        print(f"Training {args.algo} on {args.data} data...")
        
        if args.algo == 'knn':
            model = knn_train(X_train, y_train)
            save_model(model, args.algo, args.data, date)
        
        elif args.algo == 'logistic':
            weights, bias, losses = logistic_regression_train(
                X_train, y_train, args.learning_rate, args.epochs, args.batch_size
            )
            model = (weights, bias)
            save_model(model, args.algo, args.data, date)
            save_result(losses, args.algo, args.data, date, 'loss')
        
        elif args.algo == 'linear':
            weights, bias = linear_regression_train(
                X_train, y_train, method='least_squares'
            )
            losses = []
            model = (weights, bias)
            save_model(model, args.algo, args.data, date)
            save_result(losses, args.algo, args.data, date, 'loss')
        
        elif args.algo == 'svm':
            weights, bias = svm_train(
                X_train, y_train, args.learning_rate, 0.01, args.epochs
            )
            model = (weights, bias)
            save_model(model, args.algo, args.data, date)

        elif args.algo == 'ann':
            if args.data == 'house':
                X_train_scaled, x_mean, x_std = standardize_fit_transform(X_train)
                X_test_scaled = standardize_transform(X_test, x_mean, x_std)

                y_mean = np.mean(y_train)
                y_std = np.std(y_train)
                if y_std == 0:
                    y_std = 1.0
                y_train_scaled = (y_train - y_mean) / y_std
                y_test_scaled = (y_test - y_mean) / y_std

                ann_model, train_losses, val_losses = ann_train_regression(
                    X_train_scaled, y_train_scaled,
                    X_test_scaled, y_test_scaled,
                    learning_rate=args.learning_rate, epochs=args.epochs,
                    batch_size=args.batch_size or 32
                )

                model = (ann_model, x_mean, x_std, y_mean, y_std)
                save_model(model, args.algo, args.data, date)
                save_result({'train': train_losses, 'val': val_losses}, args.algo, args.data, date, 'loss')

                mse, r2, _ = ann_test_regression(X_test_scaled, y_test_scaled, ann_model)
                print(f"Validation MSE (scaled): {mse:.4f}")
                print(f"Validation R2: {r2:.4f}")
                save_result(r2, args.algo, args.data, date, 'accuracy')

            elif args.data == 'titanic':
                ann_train_titanic(
                    model_path=os.path.join('models', f'ann_titanic_{date}.pkl'),
                    loss_path=os.path.join('results', f'ann_titanic_{date}_loss.jpg')
                )

            elif args.data == 'cifar10':
                ann_train_cifar10(
                    model_path=os.path.join('models', f'ann_cifar10_{date}.pkl'),
                    loss_path=os.path.join('results', f'ann_cifar10_{date}_loss.jpg')
                )
        
        print("Training completed!")
    
    elif args.process == 'test':
        print(f"Testing {args.algo} on {args.data} data...")
        
        try:
            model = load_model(args.algo, args.data, date)
        except FileNotFoundError:
            print("Model not found. Please train first.")
            return
        
        if args.algo == 'knn':
            X_train_model, y_train_model = model
            accuracy, y_pred = knn_test(X_train_model, y_train_model, X_test, y_test, args.k)
        
        elif args.algo == 'logistic':
            weights, bias = model
            accuracy, y_pred = logistic_regression_test(X_test, y_test, weights, bias)
        
        elif args.algo == 'linear':
            weights, bias = model
            mse, y_pred = linear_regression_test(X_test, y_test, weights, bias)
            print(f"Mean Squared Error: {mse:.4f}")
            accuracy = None
        
        elif args.algo == 'svm':
            weights, bias = model
            accuracy, y_pred = svm_test(X_test, y_test, weights, bias)

        elif args.algo == 'ann':
            if args.data == 'house':
                ann_model, x_mean, x_std, y_mean, y_std = model
                X_test_scaled = standardize_transform(X_test, x_mean, x_std)
                y_test_scaled = (y_test - y_mean) / y_std
                mse, r2, y_pred = ann_test_regression(X_test_scaled, y_test_scaled, ann_model)
                print(f"Mean Squared Error (scaled): {mse:.4f}")
                print(f"R2 Score: {r2:.4f}")
                accuracy = r2
            elif args.data == 'titanic':
                loss_path = os.path.join('results', f'ann_titanic_{date}_loss.jpg')
                accuracy_path = os.path.join('results', 'accuracy.txt')
                loss, accuracy, y_pred = ann_test_titanic(
                    model_path=os.path.join('models', f'ann_titanic_{date}.pkl'),
                    loss_path=loss_path,
                    accuracy_path=accuracy_path
                )
                print(f"Loss: {loss:.4f}")
            elif args.data == 'cifar10':
                loss_path = os.path.join('results', f'ann_cifar10_{date}_loss.jpg')
                accuracy_path = os.path.join('results', 'accuracy.txt')
                loss, accuracy, y_pred = ann_test_cifar10(
                    model_path=os.path.join('models', f'ann_cifar10_{date}.pkl'),
                    loss_path=loss_path,
                    accuracy_path=accuracy_path
                )
                print(f"Loss: {loss:.4f}")
        
        if accuracy is not None:
            print(f"Accuracy: {accuracy:.4f}")
            if not (args.algo == 'ann' and args.data in ['titanic', 'cifar10']):
                save_result(accuracy, args.algo, args.data, date, 'accuracy')
        
        print("Testing completed!")

if __name__ == '__main__':
    main()