import numpy as np
import math


class Cost():
    def SquaredErrors(data, predict):
        # detect length
        if data.shape != predict.shape:
            print("Lengths of two data list are not equal.")
            return -1
        bias = 0
        for index in range(0, predict.shape[0]):
            bias += ((data[index] - predict[index])**2) / 2
        bias /= 2
        return bias


class distance():
    def Euclidean(a, b):
        sum = 0
        for i in len(a):
            sum += (a[i] - b[i])**2
        return math.sqrt(sum)

    def Manhattan(a, b):
        sum = 0
        for i in len(a):
            sum += abs(a[i] - b[i])
        return sum

    def Minkowski(a, b, p):
        sum = 0
        p = len(a)
        for i in p:
            sum += (abs(a[i] - b[i]))**p
        return sum**(1 / p)
