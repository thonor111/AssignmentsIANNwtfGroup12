if __name__ == '__main__':
    square_list = [x ** 2 for x in range(1,101)]
    print(square_list)
    even_square_list = [x ** 2 for x in range(1,101) if x ** 2 % 2 == 0]
    print(even_square_list)