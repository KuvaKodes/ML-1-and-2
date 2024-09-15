import numpy as np


train_path = "diabetes_training.csv"
test_path = "diabetes_test.csv"

# Initialize an empty list to store the data
train_data = []

# Open the CSV file in read mode
with open(train_path, 'r') as file:
    # Read the file line by line
    for line in file:
        # Split each line into individual fields using a comma as the delimiter
        fields = line.strip().split(',')
        train_data.append(fields)

test_data=[]
with open(test_path, 'r') as file:
    # Read the file line by line
    for line in file:
        # Split each line into individual fields using a comma as the delimiter
        fields = line.strip().split(',')
        test_data.append(fields)

attribute_names = train_data[0]
a = train_data[1:]
training_data = [row[:-1] for row in a]
training_labels = [row[-1] for row in a]
b = test_data[1:]
testing_data = [row[:-1] for row in b]
testing_labels = [row[-1] for row in b]


def calculate_class_probabilities(data, labels):
    # Calculate the prior probabilities of each class
    class_probabilities = {}
    total_samples = len(labels)
    
    for label in labels:
        class_probabilities[label] = labels.count(label) / total_samples

    # Calculate conditional probabilities for each attribute given the class
    attribute_probabilities = {}
    
    for i in range(len(data[0])):
        attribute_probabilities[i] = {}
        for attribute_value in set(x[i] for x in training_data):
            for label in set(labels):
                count_attr_and_class = 0
                count_class = 0

                for j in range(len(data)):
                    if data[j][i] == attribute_value and labels[j] == label:
                        count_attr_and_class += 1
                    if labels[j] == label:
                        count_class += 1
                
                # Calculate conditional probability P(attribute_value | class)
                if label not in attribute_probabilities[i]:
                    attribute_probabilities[i][label] = {}
                attribute_probabilities[i][label][attribute_value] = count_attr_and_class / count_class

    return class_probabilities, attribute_probabilities

class_probs, attr_probs = calculate_class_probabilities(training_data, training_labels)
# print(class_probs)
# print(attr_probs)

def naive_bayes_classifier(class_probs, attr_probs, instance):
    best_prob = 0
    best_class = ""
    for class_label in class_probs:
        prob = 1
        for i in range(len(instance)):
            if instance[i] in attr_probs[i][class_label]:
                prob *= attr_probs[i][class_label][instance[i]]
        prob *= class_probs[class_label]
        if prob > best_prob:
            best_prob = prob
            best_class = class_label
    return best_class


total1 = 0
num_correct1 = 0
nb_predictions = []
for i in range(len(testing_data)):
    total1+=1
    predict = naive_bayes_classifier(class_probs, attr_probs, testing_data[i])
    nb_predictions.append(predict)
    if predict==testing_labels[i]:
        num_correct1+=1
accuracy1 = num_correct1/total1
print(f"Accuracy: {accuracy1:.2f}%")

conf_matrix = np.zeros((2, 2), dtype=int)
for i in range(len(testing_labels)):
    act_label = 1 if testing_labels[i] == 'tested_positive' else 0
    pred_label = 1 if nb_predictions[i] == 'tested_positive' else 0
    conf_matrix[act_label][pred_label] += 1

print("Confusion Matrix for Naive Bayes Classifier:")
print(conf_matrix)