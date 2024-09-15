import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from collections import Counter

# Define a Node class for the decision tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Calculate entropy for a given set of class labels
def entropy(labels):
    class_counts = Counter(labels)
    probs = [class_count / len(labels) for class_count in class_counts.values()]
    entropy_val = -np.sum(p * np.log2(p) for p in probs)
    return entropy_val

# Calculate information gain for a given split
def information_gain(left_labels, right_labels, parent_labels):
    left_weight = len(left_labels) / len(parent_labels)
    right_weight = len(right_labels) / len(parent_labels)
    gain = entropy(parent_labels) - (left_weight * entropy(left_labels) + right_weight * entropy(right_labels))
    return gain

# Split data based on a given feature and threshold
def split_data(X, y, feature_index, threshold):
    left_indices = np.where(X[:, feature_index] <= threshold)[0]
    right_indices = np.where(X[:, feature_index] > threshold)[0]
    left_data, left_labels = X[left_indices], y[left_indices]
    right_data, right_labels = X[right_indices], y[right_indices]
    return left_data, left_labels, right_data, right_labels

# Build the decision tree recursively
def build_tree(X, y, feature_names, depth=0, max_depth=None):
    if len(np.unique(y)) == 1 or (max_depth is not None and depth == max_depth):
        return Node(value=np.bincount(y).argmax())

    num_features = X.shape[1]
    best_gain = 0
    best_feature_index = None
    best_threshold = None

    for feature_index in range(num_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_data, left_labels, right_data, right_labels = split_data(X, y, feature_index, threshold)
            gain = information_gain(left_labels, right_labels, y)
            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold
                best_left_data, best_left_labels = left_data, left_labels
                best_right_data, best_right_labels = right_data, right_labels

    if best_gain > 0:
        left_subtree = build_tree(best_left_data, best_left_labels, feature_names, depth + 1, max_depth)
        right_subtree = build_tree(best_right_data, best_right_labels, feature_names, depth + 1, max_depth)
        return Node(feature_index=best_feature_index, threshold=best_threshold, left=left_subtree, right=right_subtree)
    else:
        return Node(value=np.bincount(y).argmax())

# Function to predict class labels for new samples
def predict(tree, X):
    predictions = []
    for sample in X:
        node = tree
        while node.left:
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        predictions.append(node.value)
    return np.array(predictions)

# Random Forest Classifier
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y, feature_names):
        for _ in range(self.n_estimators):
            tree = build_tree(X, y, feature_names, max_depth=self.max_depth)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.trees)))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = predict(tree, X)
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions])

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
random_forest = RandomForest(n_estimators=10, max_depth=3)
random_forest.fit(X_train, y_train, feature_names)

# Make predictions on the test set
predictions = random_forest.predict(X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Display sample instances and their classifications
num_samples = 5
sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
print("\nSample Instances and their Classifications:")
for i, idx in enumerate(sample_indices):
    sample = X_test[idx]
    true_label = y_test[idx]
    pred_label = predictions[idx]
    print(f"Sample {i + 1}: Features - {sample}, True Class: {true_label}, Predicted Class: {pred_label}")

# Visualization of a single decision tree within the Random Forest
def plot_tree(tree, feature_names):
    def plot_tree_helper(node, depth=0):
        indent = "  " * depth
        if node.value is not None:
            print(f"{indent}Class: {node.value}")
        else:
            feature_name = feature_names[node.feature_index]
            print(f"{indent}if {feature_name} <= {node.threshold}")
            plot_tree_helper(node.left, depth + 1)
            print(f"{indent}else:  # {feature_name} > {node.threshold}")
            plot_tree_helper(node.right, depth + 1)

    plot_tree_helper(tree)

# Displaying the decision trees of the Random Forest
print("\nDecision Trees in the Random Forest:")
for i, tree in enumerate(random_forest.trees):
    print(f"\nTree {i + 1}:")
    plot_tree(tree, feature_names)
