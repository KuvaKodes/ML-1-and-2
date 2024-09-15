import numpy as np

#Error rate to train model on specific attribute
def calc_err_rate(attr_idx, rule, train_data, train_labels):
    total_count = 0
    err_count = 0
    for i in range(len(train_data)):
        if rule[train_data[i][attr_idx]] != train_labels[i]:
            err_count += 1
        total_count += 1
    if total_count == 0:
        return 0
    else:
        return err_count / total_count

#Algoithm logic
def one_r_algo(train_data, train_labels):
    rule_set = []
    min_err = 1
    for attr_idx in range(len(train_data[0])):
        rule = {}
        unique_attr_vals = set([item[attr_idx] for item in train_data])
        for value in unique_attr_vals:
            class_freq = {}
            freq_class = [0, 0]
            for i in range(len(train_data)):
                if train_data[i][attr_idx] == value:
                    if train_labels[i] in class_freq:
                        class_freq[train_labels[i]] += 1
                    else:
                        class_freq[train_labels[i]] = 1
                    if class_freq[train_labels[i]] > freq_class[1]:
                        freq_class = [train_labels[i], class_freq[train_labels[i]]]
            rule[value] = freq_class[0]
        error = calc_err_rate(attr_idx, rule, train_data, train_labels)
        if error < min_err:
            min_err = error
            rule_set = [attr_idx, rule]
    return rule_set

np.random.seed(0)

#Path stuff
tr_path = "diabetes_training.csv"
te_path = "diabetes_test.csv"

tr_data = []

with open(tr_path, 'r') as file:
    for line in file:
        fields = line.strip().split(',')
        tr_data.append(fields)

te_data = []
with open(te_path, 'r') as file:
    for line in file:
        fields = line.strip().split(',')
        te_data.append(fields)

#Create arrays for the necessary stuff
attr_names = tr_data[0]
a = tr_data[1:]
train_data = [row[:-1] for row in a]
train_labels = [row[-1] for row in a]
b = te_data[1:]
test_data = [row[:-1] for row in b]
test_labels = [row[-1] for row in b]

rules = one_r_algo(train_data, train_labels)

def one_r_class(rule, instance):
    return rule[1][instance[rule[0]]]

#Accuracy
num_corr = 0
total = 0
oner_pred = []
for i in range(len(test_data)):
    predict = one_r_class(rules, test_data[i])
    oner_pred.append(predict)
    if predict == test_labels[i]:
        num_corr += 1
    total += 1
accuracy = num_corr/total
print(f"Accuracy: {accuracy:.2f}%")

#Make confusion matrix from scratch
conf_matrix = np.zeros((2, 2), dtype=int)
for i in range(len(test_labels)):
    act_label = 1 if test_labels[i] == 'tested_positive' else 0
    pred_label = 1 if oner_pred[i] == 'tested_positive' else 0
    conf_matrix[act_label][pred_label] += 1

print("Confusion Matrix for OneR Classifier:")
print(conf_matrix)
