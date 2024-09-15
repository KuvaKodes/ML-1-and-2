from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

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

import numpy as np
X_train = np.array(tr_dt)
y_train = np.array(tr_lbls)
X_test = np.array(te_dt)
y_test = np.array(te_lbls)


combined_data = np.vstack((X_train, X_test))
# Convert data to integer-encoded form
data_encoder = [LabelEncoder() for _ in range(len(combined_data[0]))]

for i in range(len(combined_data[0])):
    data_encoder[i].fit([row[i] for row in combined_data])
    for j in range(len(X_train)):
        X_train[j][i] = data_encoder[i].transform([X_train[j][i]])[0]
    for j in range(len(X_test)):
        X_test[j][i] = data_encoder[i].transform([X_test[j][i]])[0]


# Create and train the Categorical Naive Bayes classifier
nb_classifier = CategoricalNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the testing data
test_predictions = nb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, test_predictions)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, test_predictions)
print("Classification Report:\n", report)

confusion_matrix_nb = confusion_matrix(y_test, test_predictions)

# Print the confusion matrix
print("Confusion Matrix for Naive Bayes Classifier:")
print(confusion_matrix_nb)