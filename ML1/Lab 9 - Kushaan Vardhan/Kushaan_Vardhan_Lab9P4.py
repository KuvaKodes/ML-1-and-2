import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_value = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k_value)

knn_classifier.fit(X_train, y_train)

predicted_labels = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy of KNN with k={k_value}: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))
