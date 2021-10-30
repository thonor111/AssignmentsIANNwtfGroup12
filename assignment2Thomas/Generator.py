import numpy as np

def logical_dataset_generator (function, length):
    for i in range(length):
        x_1 = np.random.randint(0,2)
        x_2 = np.random.randint(0,2)
        yield np.array([x_1, x_2, function(x_1, x_2)])



if __name__ == '__main__':
    and_function = lambda x_1, x_2: x_1 and x_2
    or_function = lambda x_1, x_2: x_1 or x_2
    nand_function = lambda x_1, x_2: not (x_1 and x_2)
    nor_function = lambda x_1, x_2: not (x_1 or x_2)
    xor_function = lambda x_1, x_2: (x_1 or x_2) and (not (x_1 and x_2))

    number_entries = 20
    function = xor_function
    and_generator = logical_dataset_generator(function, number_entries)
    for data_point in and_generator:
        print(data_point)