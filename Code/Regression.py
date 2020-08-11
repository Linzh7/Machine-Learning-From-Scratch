import numpy as np
import math
from Cost import Cost


class Enumerate():
    def LinerRegressionEnumerate(data_x, data_y, maxRange=10, stepSize=0.1):
        cost = costSave = 0x7fffffff
        kSave = 0
        bSave = 0
        for k in np.arange(0, maxRange, stepSize):
            for b in np.arange(0, maxRange, stepSize):
                predict_y = np.array(data_x * k + b)
                cost = Cost.SquaredErrors(data_y, predict_y)
                if cost < costSave:
                    kSave = k
                    bSave = b
                    costSave = cost
                    #print(k, b, cost)
        return kSave, bSave, costSave


class LeastSquaresMethod():
    def LinerLeastSquaresMethod(data_x, data_y):
        xMean = np.mean(data_x)
        sumYX = 0
        sumX2 = 0
        for i in np.arange(len(data_x)):
            x = data_x[i]
            y = data_y[i]
            sumYX += y * (x - xMean)
            sumX2 += x**2
        k = sumYX / (sumX2 - len(data_x) * (xMean**2))
        sumDelta = 0
        for i in np.arange(len(data_x)):
            x = data_x[i]
            y = data_y[i]
            sumDelta += (y - k * x)
        b = sumDelta / len(data_x)
        return k, b


class GradientDescent():
    def LinerGradientDescent(data_x, data_y, alpha=0.001, error=0.001):
        count = 0
        theta = [np.random.randint(100), np.random.randint(100)]
        diff = [0, 0]
        dataCost = Cost.SquaredErrors(data_x, data_y)
        predictCost = 0x7fffffff
        while predictCost - dataCost > error:
            count += 1
            for index in range(len(data_x)):
                for i in range(len(theta)):
                    for j in range(len(theta)):
                        if data_x[index] != 0 or i != 0:
                            diff[i] += (theta[j] * data_x[index]**i)
                for i in range(len(theta)):
                    theta[i] -= alpha * diff[i]
                predict_y = np.array(map(lambda x: theta0 + theta1 * x,
                                         data_x))
                predictCost = Cost.SquaredErrors(data_x, predict_y)
                if count % 100 == 0:
                    print(count, predictCost)
        return theta, predictCost