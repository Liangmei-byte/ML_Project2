import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.where(linear_output >= 0, 1, 0)

def svm_train(X, y, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
    model = SVM(learning_rate, lambda_param, n_iters)
    model.fit(X, y)
    return model.weights, model.bias

def svm_predict(X, weights, bias):
    linear_output = np.dot(X, weights) - bias
    return np.where(linear_output >= 0, 1, 0)

def svm_test(X, y, weights, bias):
    y_pred = svm_predict(X, weights, bias)
    accuracy = np.mean(y_pred == y)
    return accuracy, y_pred