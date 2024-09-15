matrix = [
    0, 0, 1, 0, 0,
    0, 1, 1, 1, 0, 
    1, 1, 0, 1, 1,
    1, 1, 0, 1, 1,
    0, 1, 1, 1, 0,
    0, 0, 1, 0, 0
]
filter1 = [1,-1,-1,
           -1,1,-1,
           -1,-1,1]
filter2 = [1,-1,1,
           -1,1,-1,
           1,-1,1]
filter3 = [-1,-1,1,
           -1,1,-1,
           1,-1,-1]
def dotProduct(v1, v2):
    answer = 0
    for i in range(len(v1)):
        answer += v1[i]*v2[i]
    return answer
def performConvolution(matrix, filter):
    new_matrix = []
    for i in range(7):
        for j in range(7):
            box=[]
            curIndex = 9*i + j
            box += matrix[curIndex:curIndex+3]
            box += matrix[curIndex+9:curIndex+12]
            box += matrix[curIndex+18:curIndex+21]
            # print(box)
            # print(filter)
            new_matrix.append(round(dotProduct(box, filter)/9, 2))

    return new_matrix

def printMatrix(matrix, n):
    for i in range(n):
        row = []
        for j in range(n):
            curIndex = n*i + j
            row.append(matrix[curIndex])
        print(row)
step1 = performConvolution(matrix, filter1)
# print(step1)
printMatrix(step1, 7)