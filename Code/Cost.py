import math

class Cost:
    def LinerRegressionCost(data, predict, k, b):
        bias = 0
        for index in range(0, len(data)):
            bias += ((data[index] - predict[index])**2) / 2
        bias /= 2
        return bias
