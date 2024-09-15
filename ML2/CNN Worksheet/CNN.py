import numpy as np

def relu(a):
    if a > 0:
        return a
    else:
        return 0


def pad_array(array):
    if array.shape[0] %2 != 0:
        array = np.pad(array, ((0, 1), (0, 0)), mode='constant')
    if array.shape[1] %2 != 0:
        array = np.pad(array, ((0, 0), (0, 1)), mode='constant')
    return array

def max_pool(array):
    array = pad_array(array)
    layer_height, layer_width = array.shape
    output_height = layer_height//2
    output_width = layer_width //2
    output_array = np.zeros((output_height, output_width))

    for i in range(0, layer_height, 2):
        for j in range(0, layer_width, 2):
            output_array[i // 2, j // 2] = np.max(array[i:i+2, j:j+2])
    return output_array

# Input matrix
input_matrix = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1, 1, -1, -1, -1, -1, -1, 1, -1],
                         [-1, -1, 1, -1, -1, -1, 1, -1, -1],
                         [-1, -1, -1, 1, -1, 1, -1, -1, -1],
                         [-1, -1, -1, -1, 1, -1, -1, -1, -1],
                         [-1, -1, -1, 1, -1, 1, -1, -1, -1],
                         [-1, -1, 1, -1, -1, -1, 1, -1, -1],
                         [-1, 1, -1, -1, -1, -1, -1, 1, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1]])

# Filter matrix
filter_matrix_1 = np.array([[1, -1, -1],
                          [-1, 1, -1],
                          [-1, -1, 1]])
filter_matrix_2 = np.array([[1, -1, 1],
                          [-1, 1, -1],
                          [1, -1, 1]])
filter_matrix_3 = np.array([[-1, -1, 1],    
                          [-1, 1, -1],
                          [1, -1, -1]])
# Stride
stride = 1

# Output matrix dimensions
output_rows = input_matrix.shape[0] - filter_matrix_1.shape[0] + 1
output_cols = input_matrix.shape[1] - filter_matrix_1.shape[1] + 1

# Initialize output matrix
output_matrix_1 = np.zeros((output_rows, output_cols))
output_matrix_2 = np.copy(output_matrix_1)
output_matrix_3 = np.copy(output_matrix_1)
# Convolve
for i in range(output_rows):
    for j in range(output_cols):
        output_matrix_1[i, j] = np.sum(input_matrix[i:i+filter_matrix_1.shape[0], j:j+filter_matrix_1.shape[1]] * filter_matrix_1)
for i in range(output_rows):
    for j in range(output_cols):
        output_matrix_2[i, j] = np.sum(input_matrix[i:i+filter_matrix_2.shape[0], j:j+filter_matrix_2.shape[1]] * filter_matrix_2)
for i in range(output_rows):
    for j in range(output_cols):
        output_matrix_3[i, j] = np.sum(input_matrix[i:i+filter_matrix_3.shape[0], j:j+filter_matrix_3.shape[1]] * filter_matrix_3)

output_matrix_1, output_matrix_2, output_matrix_3 = output_matrix_1/9, output_matrix_2/9, output_matrix_3/9

output_matrix_1 = np.round(output_matrix_1, 2)
output_matrix_2 = np.round(output_matrix_2, 2)
output_matrix_3 = np.round(output_matrix_3, 2)
#print("Output Matrix:")
#print(output_matrix_3)

relu_vectorized = np.vectorize(relu)

output_matrix_1, output_matrix_2, output_matrix_3 = relu_vectorized(output_matrix_1), relu_vectorized(output_matrix_2), relu_vectorized(output_matrix_3)
#print(output_matrix_1)
#print(pad_array(output_matrix_1))
output_matrix_1, output_matrix_2, output_matrix_3 = max_pool(output_matrix_1), max_pool(output_matrix_2), max_pool(output_matrix_3)

print(output_matrix_3)

output_matrix_1, output_matrix_2, output_matrix_3 = max_pool(output_matrix_1), max_pool(output_matrix_2), max_pool(output_matrix_3)
#print(output_matrix_3)

final = np.concatenate((output_matrix_1.ravel(), output_matrix_2.ravel(), output_matrix_3.ravel()))
#print(final)