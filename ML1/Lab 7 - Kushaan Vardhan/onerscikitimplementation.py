import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter

# Load training and test data
tr_pth = "diabetes_training.csv"
te_pth = "diabetes_test.csv"

tr_data = []
with open(tr_pth, 'r') as f:
    for line in f:
        fields = line.strip().split(',')
        tr_data.append(fields)

te_data = []
with open(te_pth, 'r') as f:
    for line in f:
        fields = line.strip().split(',')
        te_data.append(fields)

attr_names = tr_data[0]
a = tr_data[1:]
tr_dt = [row[:-1] for row in a]
tr_lbls = [row[-1] for row in a]
b = te_data[1:]
te_dt = [row[:-1] for row in b]
te_lbls = [row[-1] for row in b]

class OneRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.rule = None

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("Input features and labels must have the same length.")
        best_err = float('inf')
        for col in range(X.shape[1]):
            unique_vals = set(X[:, col])
            for val in unique_vals:
                mask = X[:, col] == val
                preds = y[mask]
                most_common = Counter(preds).most_common(1)[0][0]
                err = len(preds[preds != most_common]) / len(preds)
                if err < best_err:
                    best_err = err
                    self.rule = (col, val, most_common)

    def predict(self, X):
        if self.rule is None:
            raise NotFittedError("The classifier has not been fitted yet.")
        col, val, pred = self.rule
        return [pred] * len(X)

    def score(self, X, y):
        preds = self.predict(X)
        return sum(preds == y) / len(y)

X_tr = np.array(tr_dt)
y_tr = np.array(tr_lbls)
X_te = np.array(te_dt)
y_te = np.array(te_lbls)

oneR = OneRClassifier()
oneR.fit(X_tr, y_tr)

oneR_preds = oneR.predict(X_te)
acc = oneR.score(X_te, y_te)
print(f"Accuracy: {acc:.2f}")

# Calculate the confusion matrix manually
cm_oneR = np.zeros((2, 2), dtype=int)
for i in range(len(y_te)):
    act_label = 1 if y_te[i] == 'tested_positive' else 0
    pred_label = 1 if oneR_preds[i] == 'tested_positive' else 0
    cm_oneR[act_label][pred_label] += 1

print("Confusion Matrix for One-R Classifier:")
print(cm_oneR)
