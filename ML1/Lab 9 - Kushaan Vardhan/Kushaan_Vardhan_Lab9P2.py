from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k_value = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k_value)

knn_classifier.fit(X_train, y_train)

predicted_labels = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy of KNN with k={k_value}: {accuracy * 100:.2f}%")
