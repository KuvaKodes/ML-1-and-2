import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists to store k-values and corresponding accuracies
k_values = []
accuracies = []

# Loop through different values of k
for k in range(1, 121):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    predicted_labels = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_labels)
    k_values.append(k)
    accuracies.append(accuracy)

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title('K-value vs Accuracy for KNN on Iris dataset')
plt.xlabel('K-value')
plt.ylabel('Accuracy')
plt.xticks(range(0, 121, 10))
plt.grid(True)
plt.show()
