
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=50, random_state=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.errors_ = []
        self.weights_ = None

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
# def A(n): 
#     return 1 / (1 + np.e**n)
# new_A = np.vectorize(A)

def generate_data(num_points, coeff_line):
    X = np.random.rand(num_points, 2) * 20 - 10 
    y = np.where(X[:, 1] > coeff_line[0]*X[:, 0] + coeff_line[1], 1, 0)  
    return X, y

def plot_data(X, y, line_coeff, weights=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', marker='o', label='Below')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='x', label='Above')
    plt.plot([-10, 10], [-10 * line_coeff[0] + line_coeff[1], 10 * line_coeff[0] + line_coeff[1]], color='black', linestyle='-', linewidth=2)    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

learning_rate = 0.01
n_iter = 10
num_points = 100
coeff_line = [1, 0]  
X, y = generate_data(num_points, coeff_line)
ppn = Perceptron(learning_rate=learning_rate, n_iter=n_iter)
ppn.fit(X, y)
plot_data(X, y, coeff_line, ppn.weights_)
X_new, y_new = generate_data(20, coeff_line)
y_pred = ppn.predict(X_new)
misclassified = np.sum(y_new != y_pred)
plot_data(X_new, y_new, coeff_line)
print(f"Number of misclassifications: {misclassified}")


