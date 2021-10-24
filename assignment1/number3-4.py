import numpy as np


if __name__ == '__main__':
    arr = np.random.normal(size=(5,5))
    square_or_fourty_two = lambda x: x * x if x > 0.09 else 42.0
    for row_index in range(arr.shape[0]):
        for column_index in range(arr.shape[1]):
            arr[row_index, column_index] = square_or_fourty_two(arr[row_index, column_index])
    print(arr[:,3])