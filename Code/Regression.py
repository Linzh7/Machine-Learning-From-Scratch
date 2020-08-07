import numpy as np
import math
from Cost import Cost

class Enumerate():
    def LinerRegressionEnumerate(data_x, data_y, maxRange=100, stepSize=1):
        cost = costSave = 0x7fffffff
        kSave = 0
        bSave = 0
        for k in range(0, maxRange, stepSize):
            for b in range(0, maxRange, stepSize):
                predict_y = list(map(lambda x: x * k + b, data_x))
                cost = Cost.SquaredErrors(data_y, predict_y, k, b)
                if cost < costSave:
                    kSave = k
                    bSave = b
                    costSave = cost
                    #print(k, b, cost)
        return kSave, bSave, costSave

class LeastSquaresMethod():
    def LinerLeastSquaresMethod(data_x, data_y):
    xMean=np.mean(data_x)
    sumYX = 0
    sumX2 = 0
    for i in np.arange(len(data_x)):
        x = data_x[i]
        y = data_y[i]
        sumYX += y * (x - xMean)
        sumX2 += x**2
    w = sumYX / (sumX2 - M * (xMean ** 2))
    sumDelta = 0
    for i in np.arange(len(data_x)):
        x = data_x[i]
        y = data_y[i]
        sumDelta += (y - w * x)
    b = sumDelta / len(data_x)
    return w, b


class GradientDescent():
    def LinerGradientDescent(alpha=0.001, data_x, data_y):
        count = 0
        theta = [0]
        '''
        for i in np.arange(thetaNum):
            theta.append(0)
        '''
        while ():
            count += 1
            for index in np.arange(len(data_x)):
                diff = (theta[0] + theta[1] * data_x) - data_y
                for i in np.arange(len(theta)):
                    theta[i] -= alpha * diff * x[index][i]
                predict_y = np.array(map((lambda x: theta0 + theta1 * x, data_x))
                cost=Cost.SquaredErrors(data_x, predict_y)
                print(cost)
        return cost

