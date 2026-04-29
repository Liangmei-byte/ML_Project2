import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def logistic_regression_train(X, y, learning_rate=0.01, epochs=1000, batch_size=None):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    losses = []
    
    for epoch in range(epochs):
        if batch_size is None:
            batch_indices = np.arange(n_samples)
        else:
            batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        linear_model = np.dot(X_batch, weights) + bias
        y_pred = sigmoid(linear_model)
        
        dw = (1 / len(X_batch)) * np.dot(X_batch.T, (y_pred - y_batch))
        db = (1 / len(X_batch)) * np.sum(y_pred - y_batch)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if epoch % 100 == 0:
            linear_model_full = np.dot(X, weights) + bias
            y_pred_full = sigmoid(linear_model_full)
            loss = logistic_loss(y, y_pred_full)
            losses.append(loss)
    
    return weights, bias, losses

def logistic_regression_predict(X, weights, bias, threshold=0.5):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return np.where(y_pred >= threshold, 1, 0)

def logistic_regression_test(X, y, weights, bias):
    y_pred = logistic_regression_predict(X, weights, bias)
    accuracy = np.mean(y_pred == y)
    return accuracy, y_pred