import numpy as np
from itertools import product

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleNeuralNetwork:
    def __init__(self):
        self.weights_input_hidden = np.array([[20, 20], [20, 20]])  # 2x2 matrix for hidden layer
        self.bias_input_hidden = np.array([-10, -30])  # Bias for hidden layer
        self.weights_hidden_output = np.array([20, -20])  # Weights for output layer
        self.bias_hidden_output = -10  # Bias for output layer

    def feedforward(self, inputs):
        hidden = sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden)
        output = sigmoid(np.dot(hidden, self.weights_hidden_output) + self.bias_hidden_output)
        return output

    def predict(self, inputs):
        output = self.feedforward(inputs)
        return 1 if output > 0.5 else 0
nn = SimpleNeuralNetwork()
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
expected_xor = [0, 1, 1, 0]
expected_xnor = [1, 0, 0, 1]

print("XOR Truth Table")
print("Input1 | Input2 | Expected | Actual")
for i, input_pair in enumerate(inputs):
    actual_output = nn.predict(input_pair)
    print(f"   {input_pair[0]}    |    {input_pair[1]}   |    {expected_xor[i]}     |   {actual_output}")

print("\nXNOR Truth Table")
print("Input1 | Input2 | Expected | Actual")
for i, input_pair in enumerate(inputs):
    actual_output = 1 - nn.predict(input_pair)  # Negate the XOR output for XNOR
    print(f"   {input_pair[0]}    |    {input_pair[1]}   |    {expected_xnor[i]}     |   {actual_output}")

def n_input_xnor(inputs):
    # XNOR logic: True if all inputs are the same
    return 1 if all(x == inputs[0] for x in inputs) else 0

def generate_truth_table(n):
    print(f"{' | '.join(['Input' + str(i+1) for i in range(n)])} | XNOR Output")
    for inputs in product([0, 1], repeat=n):
        output = n_input_xnor(inputs)
        print(f"{' | '.join(map(str, inputs))} | {output}")

for n in range(2, 5):  # Example: Generate for 2, 3, and 4 inputs
    print(f"\n{n}-Input XNOR Truth Table")
    generate_truth_table(n)