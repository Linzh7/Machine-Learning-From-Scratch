import numpy as np
import math


class Cost():
    def SquaredErrors(data, predict):
        if len(data) != len(predict):
            print("Lengths of two data list are not equal.")
            return 0x7fffffff
        bias = 0
        for index in np.arange(0, len(data)):
            bias += ((data[index] - predict[index])**2) / 2
        bias /= 2
        return bias