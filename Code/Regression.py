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
    def LinerGradientDescent(data_x,
                             data_y,
                             epsilon=0.01,
                             learningRate=0.0001):
        count = 0
        theta = [0, 0]
        predictCost1 = -99999999
        while True:
            count += 1
            gradient = [0, 0]
            N = float(len(data_x))
            for index in range(len(data_x)):
                diff = (theta[0] + theta[1] * data_x[index]) - data_y[index]
                gradient[0] += (2 / N) * diff
                gradient[1] += (2 / N) * data_x[index] * diff
            theta[0] -= learningRate * gradient[0]
            theta[1] -= learningRate * gradient[1]
            predict_y = theta[0] + theta[1] * data_x
            predictCost0 = Cost.SquaredErrors(data_x, predict_y)
            if predictCost0 - predictCost1 < epsilon:
                return theta, predictCost0
            else:
                predictCost1 = predictCost0

    def HighOrderGradientDescent(data_x,
                                 data_y,
                                 order,
                                 epsilon=0.01,
                                 learningRate=0.0001):
        order += 1
        count = 0
        theta = []
        for i in range(order):
            theta.append(0)
        predictCost1 = -99999999
        while True:
            count += 1
            gradient = []
            for i in range(order):
                gradient.append(0)
            N = float(len(data_x))
            for index in range(len(data_x)):
                diff = (theta[0] + theta[1] * data_x[index]) - data_y[index]
                gradient[0] += (2 / N) * diff
                for i in range(1, order):
                    gradient[i] += (2 / N) * data_x[index] * diff**i
            for i in range(1, order):
                theta[i] -= learningRate * gradient[i]
            predict_y = np.zeros(data_x.shape)
            for i in range(1, order):
                predict_y += theta[0] * data_x**i
            predictCost0 = Cost.SquaredErrors(data_x, predict_y)
            if predictCost0 - predictCost1 < epsilon:
                return theta, predictCost0
            else:
                predictCost1 = predictCost0


def MultivariateGradientDescent(data_x,
                                data_y,
                                variableNum,
                                epsilon=0.01,
                                learningRate=0.0001):
    count = 0
    theta = []
    for i in range(variableNum):
        theta.append(0)
    predictCost1 = -99999999
    while True:
        count += 1
        gradient = []
        for i in range(variableNum):
            gradient.append(0)
        for index in range(len(data_y)):
            diff = 0
            for i in range(len(data_x)):
                diff += theta[i] * data_x[i][index]
            diff -= data_y[index]
            for i in range(variableNum):
                gradient[i] += (2.0 / len(data_y)) * data_x[i][index] * diff
            for i in range(variableNum):
                theta[i] -= learningRate * gradient[i]
        predict_y = np.zeros(data_y.shape)
        for i in range(variableNum):
            predict_y += theta[i] * data_x[i]
        predictCost0 = Cost.SquaredErrors(data_y, predict_y)
        if predictCost0 - predictCost1 < epsilon:
            return theta, predictCost0
        else:
            predictCost1 = predictCost0