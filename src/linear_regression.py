import numpy as np

def linear_regression_least_squares(X, y):
    X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
    weights = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
    bias = weights[0]
    weights = weights[1:]
    return weights, bias

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def linear_regression_gradient_descent(X, y, learning_rate=0.01, epochs=1000, batch_size=None):
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
        
        y_pred = np.dot(X_batch, weights) + bias
        loss = mean_squared_error(y_batch, y_pred)
        
        dw = (1 / len(X_batch)) * np.dot(X_batch.T, (y_pred - y_batch))
        db = (1 / len(X_batch)) * np.sum(y_pred - y_batch)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if epoch % 100 == 0:
            y_pred_full = np.dot(X, weights) + bias
            full_loss = mean_squared_error(y, y_pred_full)
            losses.append(full_loss)
    
    return weights, bias, losses

def linear_regression_train(X, y, method='least_squares', **kwargs):
    if method == 'least_squares':
        return linear_regression_least_squares(X, y)
    elif method == 'gradient_descent':
        return linear_regression_gradient_descent(X, y, **kwargs)
    else:
        raise ValueError("Method must be 'least_squares' or 'gradient_descent'")

def linear_regression_predict(X, weights, bias):
    return np.dot(X, weights) + bias

def linear_regression_test(X, y, weights, bias):
    y_pred = linear_regression_predict(X, weights, bias)
    mse = mean_squared_error(y, y_pred)
    return mse, y_pred