import torch
import concurrent.futures

def __tensor_multiply_portion(arr1, arr2, batch, row):
    y = len(arr1[0][0])
    z = len(arr2[0][0])
    result = [0.0 for _ in range(z)]
    
    for j in range(z):
        for k in range(y):
            result[j] += (arr1[batch][row][k] * arr2[batch][k][j])
    
    return result

def tensor_multiply(arr1, arr2, max_workers=16):
    b = len(arr1)
    x = len(arr1[0])
    
    result = [[None for _ in range(x)] for _ in range(b)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        
        for batch_idx in range(b):
            for row_idx in range(x):
                future = executor.submit(__tensor_multiply_portion, arr1, arr2, batch_idx, row_idx)
                future_to_index[future] = (batch_idx, row_idx)
        
        for future in concurrent.futures.as_completed(future_to_index):
            batch_idx, row_idx = future_to_index[future]
            result[batch_idx][row_idx] = future.result()
            
    return result

def __get_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else None
    return shape

def tensor_dot(arr1, arr2, dim):
    shape1 = __get_shape(arr1)
    shape2 = __get_shape(arr2)

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