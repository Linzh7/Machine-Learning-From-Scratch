import numpy as np
import math


class Cost():
    def SquaredErrors(data, predict):
        if data.shape != predict.shape:
            print("Lengths of two data list are not equal.")
            return -1
        bias = 0
        for index in range(0, predict.shape[0]):
            bias += ((data[index] - predict[index])**2) / 2
        bias /= 2
        return bias