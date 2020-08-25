import numpy as np
import math
import DataReader
from Cost import Cost


class GradientDescent():
    def MinibatchGradientDescent(data_x,
                                 data_y,
                                 variableNum,
                                 batchSize,
                                 epsilon=0.01,
                                 learningRate=0.001):
        # initialize
        count = 0
        theta = np.ones((variableNum, ), dtype=np.float32)
        predict_y = np.zeros((len(data_y), ), dtype=np.float32)
        predictCost1 = -99999

        # operate
        while True:
            count += 1
            diff = np.zeros((variableNum, ), dtype=np.float32)

            # shuffle the data set
            state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(state)
            np.random.shuffle(data_y)

            # compute diff
            for index in range(batchSize):
                predict_y[index] = np.dot(theta.T, data_x[index])
                for i in range(variableNum):
                    diff[i] += (predict_y[index] -
                                data_y[index]) * data_x[index][i]

            # update theta
            for i in range(variableNum):
                theta[i] -= learningRate * diff[i] / len(data_y)

            # stop condition
            predict_y = np.dot(data_x, theta.T)
            predictCost0 = Cost.SquaredErrors(data_y, predict_y)
            if abs(predictCost0 - predictCost1) < epsilon:
                return theta, predictCost0
            else:
                predictCost1 = predictCost0


data_x, data_y = DataReader.PrebuiltData.MyMultivariateGivenData()
theta, cost = GradientDescent.MinibatchGradientDescent(data_x, data_y, 2, 2)
print(theta, cost)