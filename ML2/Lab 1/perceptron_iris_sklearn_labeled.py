
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

data = load_iris()
X = data.data[:, :2] 
y = data.target
binary_filter = y < 2
X_binary = X[binary_filter]
y_binary = y[binary_filter]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_binary)

p = Perceptron(max_iter=1000, eta0=0.01, random_state=0)
p.fit(X_scaled, y_binary)
predictions = p.predict(X_scaled)
misclassified = np.where(predictions != y_binary)[0]
print(f"Misclassified samples: {misclassified}")

def plot_decision_boundary(X, y, classifier):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4)
    for class_value in np.unique(y):
        row_ix = np.where(y == class_value)
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired', label=data.target_names[class_value])
    plt.title('Scikit-Learn Perceptron Decision Boundary')
    plt.xlabel('Sepal length (standardized)')
    plt.ylabel('Sepal width (standardized)')
    plt.legend()
    plt.show()

plot_decision_boundary(X_scaled, y_binary, p) 
