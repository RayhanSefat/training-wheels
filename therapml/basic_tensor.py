import itertools
import pytest

def tensor_multiply(arr1, arr2):
    b, x, y = len(arr1), len(arr1[0]), len(arr1[0][0])
    z = len(arr2[0][0])
    result = [[[0.0 for _ in range(z)] for _ in range(x)] for _ in range(b)]
    
    for i in range(b):
        for j in range(x):
            for k in range(z):
                for l in range(y):
                    result[i][j][k] += (arr1[i][j][l] * arr2[i][l][k])

    return result

def get_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else None
    return shape

def tensor_dot(arr1, arr2, dim):
    shape1 = get_shape(arr1)
    shape2 = get_shape(arr2)

    if len(shape1) == 1:
        res = 0.0
        for i in range(shape1[0]):
            res += arr1[i] * arr2[i]

        return res
    elif len(shape1) == 2:
        if dim == 0:
            res = [0.0 for _ in range(shape1[1])]
            for i in range(shape1[0]):
                for j in range(shape1[1]):
                    res[j] += arr1[i][j] * arr2[i][j]
            
            return res
        else:
            res = [0.0 for _ in range(shape1[0])]
            for i in range(shape1[0]):
                for j in range(shape1[1]):
                    res[i] += arr1[i][j] * arr2[i][j]

            return res
    else:
        if dim == 0:
            res = [[0.0 for _ in range(shape1[2])] for _ in range(shape1[1])]
            for i in range(shape1[0]):
                for j in range(shape1[1]):
                    for k in range(shape1[2]):
                        res[j][k] += arr1[i][j][k] * arr2[i][j][k]

            return res
        else:
            res = [[0.0 for _ in range(shape1[2])] for _ in range(shape1[0])]
            for i in range(shape1[0]):
                for j in range(shape1[1]):
                    for k in range(shape1[2]):
                        res[i][k] += arr1[i][j][k] * arr2[i][j][k]

            return res