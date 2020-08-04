import numpy as np


class PrebuiltData():
    def Fx(x, a=0, b=3, c=3):
        return a * x**2 + b * x + c

    def MyData(rangeStart=0, rangeEnd=99, rangeNum=100, randomFloat=1):
        data_x = np.linspace(rangeStart, rangeEnd, rangeNum)
        random = np.random.randn(len(data_x))
        data_y = PrebuiltData.Fx(data_x) + randomFloat * random
        return data_x, data_y
