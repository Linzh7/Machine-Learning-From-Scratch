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
    def BatchGradientDescent(data_x,
                             data_y,
                             variableNum,
                             epsilon=0.01,
                             learningRate=0.0001):
        count = 0
        theta = np.ones((variableNum, ), dtype=np.float32)
        predict_y = np.zeros((len(data_y), ), dtype=np.float32)
        predictCost1 = -99999
        while True:
            count += 1
            diff = np.zeros((variableNum, ), dtype=np.float32)
            for index in range(len(data_y)):
                predict_y[index] = np.dot(theta.T, data_x[index])
                for i in range(variableNum):
                    diff[i] += (predict_y[index] -
                                data_y[index]) * data_x[index][i]

            for i in range(variableNum):
                theta[i] -= learningRate * diff[i] / len(data_y)

            predictCost0 = Cost.SquaredErrors(data_y, predict_y)
            if abs(predictCost0 - predictCost1) < epsilon:
                return theta, predictCost0
            else:
                predictCost1 = predictCost0
