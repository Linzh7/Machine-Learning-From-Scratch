import random

class PrebuiltData:
    def MyData(int rangeStart = 0, int rangeEnd = 99, rangeStep=1, int k = 3, int b = 6):
        data_x = list(range(rangeStart, rangeEnd, rangeStep))
        data_y = list(map(lambda x: x * (k - 1 + random.random()) + b * random.random(), data_x))
        return data_x, data_y
