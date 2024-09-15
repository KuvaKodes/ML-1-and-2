import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

outlier_indices = np.random.choice(len(X), 10, replace=False)  # Select 10 random indices as outliers
X[outlier_indices] += 15  # Adding outlier values to those indices

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = []
accuracies = []

for k in range(1, 121):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    predicted_labels = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_labels)
    k_values.append(k)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title('K-value vs Accuracy for KNN with Outliers')
plt.xlabel('K-value')
plt.ylabel('Accuracy')
plt.xticks(range(0, 121, 10))
plt.grid(True)
plt.show()
