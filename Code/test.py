import numpy as np
from Cost import Cost
import DataReader


def MultivariateBatchGradientDescent(data_x,
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


data_x, data_y = DataReader.PrebuiltData.MyMultivariateGivenData()
theta, cost = MultivariateBatchGradientDescent(data_x, data_y, 2)
print(theta, cost)