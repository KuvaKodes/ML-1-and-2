class Perceptron:
    def __init__(self, weights=None, bias=0):
        self.weights = weights if weights is not None else []
        self.bias = bias

    def activate(self, inputs):
        return 1 if sum(w * i for w, i in zip(self.weights, inputs)) + self.bias > 0 else 0

class MLP:
    def __init__(self):
        self.hidden_layer = [Perceptron() for _ in range(6)]
        self.output_layer = Perceptron()

    def feedforward(self, inputs):
        hidden_layer_outputs = [perceptron.activate(inputs) for perceptron in self.hidden_layer]
        return self.output_layer.activate(hidden_layer_outputs)

    def set_weights_and_biases(self, hidden_weights, hidden_biases, output_weights, output_bias):
        for i, perceptron in enumerate(self.hidden_layer):
            perceptron.weights = hidden_weights[i]
            perceptron.bias = hidden_biases[i]
        self.output_layer.weights = output_weights
        self.output_layer.bias = output_bias

    def generate_truth_table(self, expected_outputs):
        print(f"{'X1':^3} {'X2':^3} {'X3':^3} {'X4':^3} {'X5':^3} {'Y(exp)':^7} {'Y(act)':^7}")
        for i in range(32):
            inputs = [(i >> bit) & 1 for bit in range(5)]
            expected_output = expected_outputs[i]
            actual_output = self.feedforward(inputs)
            print(f"{inputs[0]:^3} {inputs[1]:^3} {inputs[2]:^3} {inputs[3]:^3} {inputs[4]:^3} {expected_output:^7} {actual_output:^7}")

mlp = MLP()

hidden_layer_weights = [
    [-1, -1, 1, 1, -1],  
    [-1, 1, -1, 1, 1],  
    [-1, 1, 1, -1, -1],  
    [1, -1, -1, -1, 1], 
    [1, -1, 1, 1, 1],  
    [1, 1, -1, -1, 1],  
]
hidden_layer_biases = [-2, -3, -2, -2, -4, -3] 


output_layer_weights = [1, 1, 1, 1, 1, 1]
output_layer_bias = -1

mlp.set_weights_and_biases(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_bias)

expected_outputs = [0] * 32
expected_outputs[6] = 1  
expected_outputs[11] = 1  
expected_outputs[12] = 1  
expected_outputs[17] = 1  
expected_outputs[22] = 1  
expected_outputs[25] = 1  

mlp.generate_truth_table(expected_outputs)


