import numpy as np
from itertools import product

class Perceptron:
    def __init__(self, num_inputs, function_type):
        self.function_type = function_type
        if function_type == 'OR':
            self.weights = np.ones(num_inputs)
            self.bias = -0.5
        elif function_type == 'AND':
            self.weights = np.ones(num_inputs)
            self.bias = -(num_inputs - 1)
        elif function_type == 'NAND':
            self.weights = -np.ones(num_inputs)
            self.bias = num_inputs
        else:
            raise ValueError("Function type must be 'OR', 'AND', or 'NAND'.")

    def activate(self, x):
        linear_output = np.dot(self.weights, x) + self.bias
        return 1 if linear_output > 0 else 0

    def calculate_expected_output(self, inputs):
        if self.function_type == 'OR':
            return 1 if any(inputs) else 0
        elif self.function_type == 'AND':
            return 1 if all(inputs) else 0
        elif self.function_type == 'NAND':
            return 0 if all(inputs) else 1

def truth_table(perceptron, num_inputs):
    print("Inputs | Expected | Actual")
    for inputs in product([0, 1], repeat=num_inputs):
        actual_output = perceptron.activate(np.array(inputs))
        expected_output = perceptron.calculate_expected_output(inputs)
        print(f"{' | '.join(map(str, inputs))} | {expected_output} | {actual_output}")

num_inputs = 4
and_perceptron = Perceptron(num_inputs, 'AND')
print("AND Function:")
truth_table(and_perceptron, num_inputs)
or_perceptron = Perceptron(num_inputs, 'OR')
print("OR Function:")
truth_table(or_perceptron, num_inputs)
nand_perceptron = Perceptron(num_inputs, 'NAND')
print("NAND Function:")
truth_table(nand_perceptron, num_inputs)