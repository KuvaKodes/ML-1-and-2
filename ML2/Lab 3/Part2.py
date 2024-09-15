import sys
import ast
import numpy as np
import math
import pickle
from scipy import ndimage
import matplotlib.pyplot as plt

accuracies = []
errors = []
wi = []

def A(n): 
    return 1 / (1 + np.e**-n)
new_A = np.vectorize(A)

#Back Propgation
def back_propogate(A_vec, weights, biases, training_set, epochs, untampered):
    l = 0.2
    for epoch in range(epochs):
        error_list = list()
        for input, actual_output in training_set:
            a = list() 
            a.append(input)
            for layer in range(1, len(weights)):
                a.append(A_vec((weights[layer]@a[layer-1])+biases[layer]))
            #Error Calcs
            error_list.append(0.5 * ((np.sum(np.square((actual_output - a[len(weights)-1])))) ** 0.5))
            delta_lists = [None] * len(weights)
            delta_lists[len(weights)-1] = (a[len(weights)-1] * (1 - a[len(weights)-1])) * (actual_output - a[len(weights)-1])
            #Backwards passes
            for layer in range(len(weights)-2, 0, -1):
                delta_lists[layer] = (a[layer] * (1-a[layer])) * (weights[layer+1].T @ delta_lists[layer+1])
            for layer in range(1,len(weights)):
                weights[layer] = weights[layer] + (l * (delta_lists[layer] @ a[layer-1].T))
                biases[layer] = biases[layer] + (l*delta_lists[layer])
        print(epoch, epoch_accuracy := 100 - accuracy_calculator(weights, biases, untampered))
        accuracies.append(epoch_accuracy)
        errors.append(sum(error_list)/len(error_list))
        wi.append(weights[1][0, 1])
        with open("save_data.pkl", "wb") as f:
            pickle.dump((weights, biases), f)        
       # l = sum(error_list)/len(error_list)
    return weights, biases

#Calculates accuracy
def accuracy_calculator(weights, biases, training_set):
    num_inaccurate = 0 
    for input, actual_output in training_set:
        truth_test = np.zeros((10, 1))
        output = p_net(new_A, weights, biases, input)
        max_val = np.max(output)
        for x in range(10):
            if output[x, 0]  != max_val:
                output[x, 0] = 0
            else:
                output[x, 0] = 1
        if not np.array_equal((actual_output - output), truth_test):
            num_inaccurate += 1
    return num_inaccurate/len(training_set) * 100

#Runs the Neural Net
def p_net(A_vec, weights, biases, inputs):
    a = list()
    a.append(inputs)
    for layer in range(1, len(weights)):
        a.append(A_vec((weights[layer]@a[layer-1])+biases[layer]))
    return a[len(weights)-1]

#Initiialzer of random weights and biases
def create_random_network(size_array):
    random_weights = [None, ]
    random_biases = [None, ]
    for index, size in enumerate(size_array):
        if index == 0: 
            random_weights.append(2 * np.random.rand(size, 784)- 1)
        else: 
            random_weights.append(2 * np.random.rand(size, size_array[index-1]) - 1)
        random_biases.append(2 * np.random.rand(size, 1) - 1)
    return random_weights, random_biases

#The below code opens the training file and formats it into an input array that is useful to me, it also adds distortions to the dataset that will improve recognition/performance by avoiding overfitting
with open("mnist_train.csv") as f:
    normed_data_set = list()
    untampered_data_set = list() 
    for line in f: 
        output = [0] * 10
        output[int(line[0])] = 1
        output = np.reshape(output, (10, 1))
        inputs = [float(val)/255 for val in line[2:].split(',')]
        inputs = np.reshape(inputs, (784, 1))
        untampered_data_set.append((inputs, output))
        distort = np.random.randint(0, 7)
        #distort = 5
        if distort == 0:
            #Normal
            final_array = [float(val)/255 for val in line[2:].split(',')]
            final_matrix = np.reshape(final_array, (784, 1))
        elif distort == 1:
            #Shift Right
            input_array = line[2:].strip().split(',')
            final_array = list()
            for x in range(28):
                row_list = np.array([float(i)/255 for i in input_array[28 * x: 28 * (x+1)]])
                row_list = np.roll(row_list, 1)
                final_array.append(row_list)
            final_matrix = np.array(final_array)
            final_matrix = np.reshape(final_matrix, (784, 1))
        elif distort == 2:
            #Shift Left
            input_array = line[2:].strip().split(',')
            final_array = list()
            for x in range(28):
                row_list = np.array([float(i)/255 for i in input_array[28 * x: 28 * (x+1)]])
                row_list = np.roll(row_list, -1)
                final_array.append(row_list)
            final_matrix = np.array(final_array)
            final_matrix = np.reshape(final_matrix, (784, 1))        
        elif distort == 3:
            #Shift Up
            input_array = line[2:].strip().split(',')
            final_array = list()
            for x in range(28):
                row_list = np.array([float(i)/255 for i in input_array[28 * x: 28 * (x+1)]])
                final_array.append(row_list)
            final_matrix = np.array(final_array)
            final_matrix = np.roll(final_matrix, -1, 0)
            final_matrix = np.reshape(final_matrix, (784, 1))           
        elif distort == 4: 
            #Shift Down
            input_array = line[2:].strip().split(',')
            final_array = list()
            for x in range(28):
                row_list = np.array([float(i)/255 for i in input_array[28 * x: 28 * (x+1)]])
                final_array.append(row_list)
            final_matrix = np.array(final_array)
            final_matrix = np.roll(final_matrix, 1, 0)
            final_matrix = np.reshape(final_matrix, (784, 1))           
        elif distort == 5:
            #Rotate Right 15 Degrees 
            input_array = line[2:].strip().split(',')
            final_array = list()
            for x in range(28):
                row_list = np.array([float(i)/255 for i in input_array[28 * x: 28 * (x+1)]])
                final_array.append(row_list)  
            final_matrix = ndimage.rotate(final_array, 15, reshape=False)
            final_matrix = np.array(final_matrix)      
            final_matrix = np.reshape(final_matrix, (784,1))    
        elif distort == 6:
            #Rotate Left 15 Degrees
            input_array = line[2:].strip().split(',')
            final_array = list()
            for x in range(28):
                row_list = np.array([float(i)/255 for i in input_array[28 * x: 28 * (x+1)]])
                final_array.append(row_list)  
            final_matrix = ndimage.rotate(final_array, -15, reshape=False)
            final_matrix = np.array(final_matrix)
            final_matrix = np.reshape(final_matrix, (784,1))
        normed_data_set.append((final_matrix, output))
    print("done distorting")

ran_weights, ran_biases = create_random_network([300,100,10])
w, b = back_propogate(new_A, ran_weights, ran_biases, normed_data_set, 20, untampered_data_set)

epochs = range(1, len(errors) + 1)

# Plotting Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, errors, label='Loss')
plt.title('Epoch vs. Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracies, label='Accuracy')
plt.title('Epoch vs. Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(wi, errors, label='Weight vs. Loss')
plt.title('Weight Value vs. Loss')
plt.xlabel('Weight Value')
plt.ylabel('Loss')
plt.legend()
plt.show()

with open("mnist_test.csv") as f1:
    normed_output_set = list() 
    for line in f1: 
        output = [0] * 10
        output[int(line[0])] = 1
        output = np.reshape(output, (10, 1))
        inputs = [float(val)/255 for val in line[2:].split(',')]
        inputs = np.reshape(inputs, (784, 1))
        normed_output_set.append((inputs, output))
# with open("save_data.pkl", "rb") as f2:
#      best_weights, best_biases = pickle.load(f2)

# print(accuracy_calculator(best_weights, best_biases, normed_output_set))


        

