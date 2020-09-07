import numpy as np
import math


def Sigmoid(x):
    y = 1 / (1 + math.e**(-x))
    return y