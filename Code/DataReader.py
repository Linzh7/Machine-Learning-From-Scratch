import random

class data:
    def MyData(int rangeA = 0, int rangeB = 99, int k = 3, int b = 6):
        data_x = list(range(rangeA, rangeB))
        data_y = list(map(lambda x: x * (k - 1 + random.random()) + b * random.random(), data_x))
        return data_x, data_y
