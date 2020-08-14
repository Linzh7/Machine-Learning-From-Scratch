import numpy as np
from Cost import Cost
import DataReader


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


data_x, data_y = DataReader.PrebuiltData.MyGivenData()
theta, cost = GradientDescent.LinerGradientDescent(data_x, data_y)
print(theta, cost)