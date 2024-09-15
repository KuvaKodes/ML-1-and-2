import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize the neural network parameters
input_size = 2
hidden_size = 2
output_size = 2

# Given weights and biases
weights_input_to_hidden = np.array([[0.15, 0.25], [0.20, 0.30]])
weights_hidden_to_output = np.array([[0.40, 0.50], [0.45, 0.55]])
biases_hidden = np.array([0.35, 0.35])
biases_output = np.array([0.60, 0.60])

# Target outputs
targets = np.array([0.01, 0.99])

# Learning rate
learning_rate = 0.5

# Number of epochs
epochs = 10000

# Store changes in total error and weight w1
error_history = []
w1_history = []

# Training process
for epoch in range(epochs):
    # Forward pass
    input_layer = np.array([0.05, 0.10])
    hidden_input = np.dot(input_layer, weights_input_to_hidden) + biases_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_to_output) + biases_output
    final_output = sigmoid(final_input)

    # Calculate error
    error = 0.5 * (targets - final_output) ** 2
    total_error = np.sum(error)
    error_history.append(total_error)

    # Backward pass
    # Output layer error
    output_error = targets - final_output
    output_delta = output_error * sigmoid_derivative(final_output)

    # Hidden layer error
    hidden_error = output_delta.dot(weights_hidden_to_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Update the weights and biases
    weights_hidden_to_output += learning_rate * np.outer(hidden_output, output_delta)
    weights_input_to_hidden += learning_rate * np.outer(input_layer, hidden_delta)
    biases_output += learning_rate * output_delta
    biases_hidden += learning_rate * hidden_delta

    # Store the history of w1 for later analysis
    w1_history.append(weights_input_to_hidden[0, 0])

    # Log progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Total Error: {total_error}')

# Plotting the total error over epochs
plt.figure(figsize=(10,5))
plt.plot(error_history, label='Total Error')
plt.title('Error Trend over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Total Error')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the weight w1 against total error
plt.figure(figsize=(10,5))
plt.scatter(w1_history, error_history, s=1)
plt.title('Weight w1 vs Total Error')
plt.xlabel('Weight w1')
plt.ylabel('Total Error')
plt.grid(True)
plt.show()

