import numpy as np
from Cost import Cost
import DataReader


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
                theta[i] -= learningRate * gradient[0]
        predict_y = np.zeros(data_y.shape)
        for i in range(variableNum):
            predict_y += theta[i] * data_x[i]
        predictCost0 = Cost.SquaredErrors(data_y, predict_y)
        if predictCost0 - predictCost1 < epsilon:
            pass
        #    return theta, predictCost0
        else:
            predictCost1 = predictCost0


data_x, data_y = DataReader.PrebuiltData.MyMultivariateGivenData()
theta, cost = MultivariateGradientDescent(data_x, data_y, 2)
print(theta, cost)