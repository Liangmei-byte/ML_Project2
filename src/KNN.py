import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    y_pred = []
    
    for test_point in X_test:
        distances = []
        
        for i, train_point in enumerate(X_train):
            distance = euclidean_distance(test_point, train_point)
            distances.append((distance, y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        
        k_nearest_labels = [label for _, label in k_nearest]
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common)
    
    return np.array(y_pred)

def knn_train(X_train, y_train):
    return X_train, y_train

def knn_test(X_train, y_train, X_test, y_test, k=3):
    y_pred = knn_predict(X_train, y_train, X_test, k)
    accuracy = np.mean(y_pred == y_test)
    return accuracy, y_pred