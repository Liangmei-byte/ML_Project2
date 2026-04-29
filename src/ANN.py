import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

try:
    from .data_preprocess import (
        extract_hog_features,
        load_cifar10_split_data,
        load_titanic_data,
        preprocess_titanic_data,
    )
except ImportError:
    from data_preprocess import (
        extract_hog_features,
        load_cifar10_split_data,
        load_titanic_data,
        preprocess_titanic_data,
    )


def standardize_fit_transform(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def standardize_transform(X, mean, std):
    return (X - mean) / std


def _relu(z):
    return np.maximum(0, z)


def _relu_grad(z):
    return (z > 0).astype(float)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def _binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def _categorical_cross_entropy(y_true_one_hot, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))


def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def _one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y.astype(int)] = 1
    return one_hot


def ann_train_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_sizes=(64, 32),
    learning_rate=0.001,
    epochs=1000,
    batch_size=32,
    seed=42,
):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X_train.shape
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    h1, h2 = hidden_sizes

    W1 = rng.normal(0, np.sqrt(2.0 / n_features), (n_features, h1))
    b1 = np.zeros((1, h1))
    W2 = rng.normal(0, np.sqrt(2.0 / h1), (h1, h2))
    b2 = np.zeros((1, h2))
    W3 = rng.normal(0, np.sqrt(1.0 / h2), (h2, 1))
    b3 = np.zeros((1, 1))

    train_losses = []
    val_losses = []

    for _ in range(epochs):
        indices = rng.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            m = X_batch.shape[0]

            z1 = X_batch @ W1 + b1
            a1 = _relu(z1)
            z2 = a1 @ W2 + b2
            a2 = _relu(z2)
            y_pred = a2 @ W3 + b3

            dy = (2.0 / m) * (y_pred - y_batch)

            dW3 = a2.T @ dy
            db3 = np.sum(dy, axis=0, keepdims=True)

            da2 = dy @ W3.T
            dz2 = da2 * _relu_grad(z2)
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ W2.T
            dz1 = da1 * _relu_grad(z1)
            dW1 = X_batch.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            W3 -= learning_rate * dW3
            b3 -= learning_rate * db3
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1

        train_pred = ann_predict_regression(X_train, (W1, b1, W2, b2, W3, b3))
        val_pred = ann_predict_regression(X_val, (W1, b1, W2, b2, W3, b3))
        train_losses.append(_mse(y_train.ravel(), train_pred))
        val_losses.append(_mse(y_val.ravel(), val_pred))

    model = (W1, b1, W2, b2, W3, b3)
    return model, train_losses, val_losses


def ann_predict_regression(X, model):
    W1, b1, W2, b2, W3, b3 = model
    a1 = _relu(X @ W1 + b1)
    a2 = _relu(a1 @ W2 + b2)
    y_pred = a2 @ W3 + b3
    return y_pred.ravel()


def ann_test_regression(X, y, model):
    y_pred = ann_predict_regression(X, model)
    mse = _mse(y, y_pred)
    r2 = _r2_score(y, y_pred)
    return mse, r2, y_pred


def ann_train_classification(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_sizes=(32, 16),
    learning_rate=0.001,
    epochs=500,
    batch_size=32,
    seed=42,
):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X_train.shape
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    h1, h2 = hidden_sizes
    W1 = rng.normal(0, np.sqrt(2.0 / n_features), (n_features, h1))
    b1 = np.zeros((1, h1))
    W2 = rng.normal(0, np.sqrt(2.0 / h1), (h1, h2))
    b2 = np.zeros((1, h2))
    W3 = rng.normal(0, np.sqrt(1.0 / h2), (h2, 1))
    b3 = np.zeros((1, 1))

    train_losses = []
    val_losses = []

    for _ in range(epochs):
        indices = rng.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            m = X_batch.shape[0]

            z1 = X_batch @ W1 + b1
            a1 = _relu(z1)
            z2 = a1 @ W2 + b2
            a2 = _relu(z2)
            z3 = a2 @ W3 + b3
            y_pred = _sigmoid(z3)

            dz3 = (y_pred - y_batch) / m
            dW3 = a2.T @ dz3
            db3 = np.sum(dz3, axis=0, keepdims=True)

            da2 = dz3 @ W3.T
            dz2 = da2 * _relu_grad(z2)
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ W2.T
            dz1 = da1 * _relu_grad(z1)
            dW1 = X_batch.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            W3 -= learning_rate * dW3
            b3 -= learning_rate * db3
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1

        train_prob = ann_predict_classification_proba(X_train, (W1, b1, W2, b2, W3, b3))
        val_prob = ann_predict_classification_proba(X_val, (W1, b1, W2, b2, W3, b3))
        train_losses.append(_binary_cross_entropy(y_train.ravel(), train_prob))
        val_losses.append(_binary_cross_entropy(y_val.ravel(), val_prob))

    model = (W1, b1, W2, b2, W3, b3)
    return model, train_losses, val_losses


def ann_predict_classification_proba(X, model):
    W1, b1, W2, b2, W3, b3 = model
    a1 = _relu(X @ W1 + b1)
    a2 = _relu(a1 @ W2 + b2)
    y_prob = _sigmoid(a2 @ W3 + b3)
    return y_prob.ravel()


def ann_predict_classification(X, model, threshold=0.5):
    y_prob = ann_predict_classification_proba(X, model)
    return (y_prob >= threshold).astype(int)


def ann_test_classification(X, y, model, threshold=0.5):
    y_prob = ann_predict_classification_proba(X, model)
    y_pred = (y_prob >= threshold).astype(int)
    loss = _binary_cross_entropy(y, y_prob)
    accuracy = _accuracy_score(y, y_pred)
    return loss, accuracy, y_pred


def ann_train_multiclass(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    hidden_sizes=(128, 64),
    learning_rate=0.001,
    epochs=30,
    batch_size=128,
    seed=42,
):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X_train.shape
    y_train_one_hot = _one_hot_encode(y_train, num_classes)
    y_val_one_hot = _one_hot_encode(y_val, num_classes)

    h1, h2 = hidden_sizes
    W1 = rng.normal(0, np.sqrt(2.0 / n_features), (n_features, h1))
    b1 = np.zeros((1, h1))
    W2 = rng.normal(0, np.sqrt(2.0 / h1), (h1, h2))
    b2 = np.zeros((1, h2))
    W3 = rng.normal(0, np.sqrt(1.0 / h2), (h2, num_classes))
    b3 = np.zeros((1, num_classes))

    train_losses = []
    val_losses = []

    for _ in range(epochs):
        indices = rng.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_one_hot[indices]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            m = X_batch.shape[0]

            z1 = X_batch @ W1 + b1
            a1 = _relu(z1)
            z2 = a1 @ W2 + b2
            a2 = _relu(z2)
            z3 = a2 @ W3 + b3
            y_pred = _softmax(z3)

            dz3 = (y_pred - y_batch) / m
            dW3 = a2.T @ dz3
            db3 = np.sum(dz3, axis=0, keepdims=True)

            da2 = dz3 @ W3.T
            dz2 = da2 * _relu_grad(z2)
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ W2.T
            dz1 = da1 * _relu_grad(z1)
            dW1 = X_batch.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            W3 -= learning_rate * dW3
            b3 -= learning_rate * db3
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1

        train_prob = ann_predict_multiclass_proba(X_train, (W1, b1, W2, b2, W3, b3))
        val_prob = ann_predict_multiclass_proba(X_val, (W1, b1, W2, b2, W3, b3))
        train_losses.append(_categorical_cross_entropy(y_train_one_hot, train_prob))
        val_losses.append(_categorical_cross_entropy(y_val_one_hot, val_prob))

    model = (W1, b1, W2, b2, W3, b3)
    return model, train_losses, val_losses


def ann_predict_multiclass_proba(X, model):
    W1, b1, W2, b2, W3, b3 = model
    a1 = _relu(X @ W1 + b1)
    a2 = _relu(a1 @ W2 + b2)
    return _softmax(a2 @ W3 + b3)


def ann_predict_multiclass(X, model):
    y_prob = ann_predict_multiclass_proba(X, model)
    return np.argmax(y_prob, axis=1)


def ann_test_multiclass(X, y, model, num_classes):
    y_prob = ann_predict_multiclass_proba(X, model)
    y_pred = np.argmax(y_prob, axis=1)
    y_one_hot = _one_hot_encode(y, num_classes)
    loss = _categorical_cross_entropy(y_one_hot, y_prob)
    accuracy = _accuracy_score(y, y_pred)
    return loss, accuracy, y_pred


def plot_loss_curve(train_losses, val_losses, save_path, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_accuracy_result(dataset_name, accuracy, loss, accuracy_path):
    os.makedirs(os.path.dirname(accuracy_path), exist_ok=True)
    with open(accuracy_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now()}: ann on {dataset_name} - Accuracy: {accuracy:.4f}, "
            f"Loss: {loss:.4f}\n"
        )


def ann_train_titanic(
    model_path=os.path.join("models", "ann_titanic.pkl"),
    loss_path=os.path.join("results", "ann_titanic_loss.jpg"),
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_df, _ = load_titanic_data()
    X, y = preprocess_titanic_data(train_df)

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    X_train_scaled, x_mean, x_std = standardize_fit_transform(X_train)
    X_val_scaled = standardize_transform(X_val, x_mean, x_std)

    model, train_losses, val_losses = ann_train_classification(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        hidden_sizes=(32, 16),
        learning_rate=0.001,
        epochs=500,
        batch_size=32,
        seed=42,
    )

    model_bundle = {
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    plot_loss_curve(train_losses, val_losses, loss_path, "ANN Titanic Loss Curve")
    return model_bundle


def ann_test_titanic(
    model_path=os.path.join("models", "ann_titanic.pkl"),
    loss_path=os.path.join("results", "ann_titanic_loss.jpg"),
    accuracy_path=os.path.join("results", "accuracy.txt"),
):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please run main.py --algo=ann --data=titanic --process=train first.")

    _, test_df = load_titanic_data()
    X_test, y_test = preprocess_titanic_data(test_df)

    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    X_test_scaled = standardize_transform(X_test, model_bundle["x_mean"], model_bundle["x_std"])
    loss, accuracy, y_pred = ann_test_classification(X_test_scaled, y_test, model_bundle["model"])

    plot_loss_curve(
        model_bundle["train_losses"],
        model_bundle["val_losses"],
        loss_path,
        "ANN Titanic Loss Curve",
    )
    save_accuracy_result("titanic", accuracy, loss, accuracy_path)
    return loss, accuracy, y_pred


def ann_train_cifar10(
    model_path=os.path.join("models", "ann_cifar10.pkl"),
    loss_path=os.path.join("results", "ann_cifar10_loss.jpg"),
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    X_train_raw, y_train_raw, _, _ = load_cifar10_split_data()
    X_train_hog = extract_hog_features(X_train_raw)

    split_idx = int(0.9 * len(X_train_hog))
    X_train, X_val = X_train_hog[:split_idx], X_train_hog[split_idx:]
    y_train, y_val = y_train_raw[:split_idx], y_train_raw[split_idx:]

    X_train_scaled, x_mean, x_std = standardize_fit_transform(X_train)
    X_val_scaled = standardize_transform(X_val, x_mean, x_std)

    model, train_losses, val_losses = ann_train_multiclass(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        num_classes=10,
        hidden_sizes=(128, 64),
        learning_rate=0.001,
        epochs=30,
        batch_size=128,
        seed=42,
    )

    model_bundle = {
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "feature_type": "hog",
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    plot_loss_curve(train_losses, val_losses, loss_path, "ANN CIFAR10 Loss Curve")
    return model_bundle


def ann_test_cifar10(
    model_path=os.path.join("models", "ann_cifar10.pkl"),
    loss_path=os.path.join("results", "ann_cifar10_loss.jpg"),
    accuracy_path=os.path.join("results", "accuracy.txt"),
):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please run main.py --algo=ann --data=cifar10 --process=train first.")

    _, _, X_test_raw, y_test = load_cifar10_split_data()
    X_test_hog = extract_hog_features(X_test_raw)

    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    X_test_scaled = standardize_transform(X_test_hog, model_bundle["x_mean"], model_bundle["x_std"])
    loss, accuracy, y_pred = ann_test_multiclass(
        X_test_scaled, y_test, model_bundle["model"], num_classes=10
    )

    plot_loss_curve(
        model_bundle["train_losses"],
        model_bundle["val_losses"],
        loss_path,
        "ANN CIFAR10 Loss Curve",
    )
    save_accuracy_result("cifar10", accuracy, loss, accuracy_path)
    return loss, accuracy, y_pred

