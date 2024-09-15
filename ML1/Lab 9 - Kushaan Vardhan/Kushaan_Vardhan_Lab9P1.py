import numpy as np
from collections import Counter

# KNN algorithm
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def knn(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = []
        for train_point, label in zip(X_train, y_train):
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, label))
        
        distances.sort(key=lambda x: x[0])  # Sort distances
        
        k_nearest_neighbors = distances[:k]  # Select the first k distances
        k_nearest_labels = [neighbor[1] for neighbor in k_nearest_neighbors]  # Get labels of k-nearest neighbors
        
        # Find the most common label in k-nearest neighbors
        predicted_label = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(predicted_label)
    
    return predictions

# Define the dtype for structured array
dtype = [('sepallength', float), ('sepalwidth', float), ('petallength', float), ('petalwidth', float), ('class', 'U10')]

# Load the Iris dataset from CSV files
file_training = "iris_training.csv"
file_testing = "iris_testing.csv"

# Load the data as a structured array with column names and specified data types
iris_train = np.genfromtxt(file_training, delimiter=',', dtype=dtype, names=True)
iris_test = np.genfromtxt(file_testing, delimiter=',', dtype=dtype, names=True)

# Extract features (X) and labels (y) from the structured arrays
X_train = np.vstack([iris_train['sepallength'], iris_train['sepalwidth'], iris_train['petallength'], iris_train['petalwidth']]).T
y_train = iris_train['class']

X_test = np.vstack([iris_test['sepallength'], iris_test['sepalwidth'], iris_test['petallength'], iris_test['petalwidth']]).T
y_test = iris_test['class']

# Choose the value of K
k_value = 3

# Make predictions using the KNN algorithm
predicted_labels = knn(X_train, y_train, X_test, k=k_value)

# Calculate accuracy
accuracy = np.mean(predicted_labels == y_test)
print(f"Accuracy of KNN with k={k_value}: {accuracy * 100:.2f}%")
