import cat
import generator


if __name__ == '__main__':
    # 3.1
    cat1 = cat.cat("Kittysaurus Rex")
    cat2 = cat.cat("Snowball IX")
    cat1.greet(cat2)

    #3.2
    square_list = [x ** 2 for x in range(1,101)]
    print(square_list)
    even_square_list = [x ** 2 for x in range(1,101) if x ** 2 % 2 == 0]
    print(even_square_list)

    # 3.3
    generator = generator.generator()
    for i in range(10):
        generator.generate()

    # 3.4
