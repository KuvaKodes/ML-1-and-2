
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from collections import Counter

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def feature_importance_rf(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model.feature_importances_

def weighted_euclidean_distance(point1, point2, weights):
    return np.sqrt(np.sum(weights * (point1 - point2) ** 2))

def knn_predict(X_train, y_train, X_test, k, weights):
    predictions = []
    for test_point in X_test:
        distances = [weighted_euclidean_distance(train_point, test_point, weights) 
                     for train_point in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    return predictions

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, cm

def run_model(file_path):
    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    weights = feature_importance_rf(X_train, y_train)
    weights = 1 / (weights + 0.0001)
    k = 10
    y_pred = knn_predict(X_train, y_train, X_test, k, weights)
    accuracy, precision, recall, cm = evaluate_model(y_test, y_pred)
    return accuracy, precision, recall, cm

if __name__ == "__main__":
    file_paths = ['Breast_cancer_data.csv', 'diabetes.csv', 'heart.csv']
    for path in file_paths:
        acc, prec, rec, cm = run_model(path)
        print(f'Results for {path}:')
        print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}, Confusion Matrix:\n{cm}\n')
