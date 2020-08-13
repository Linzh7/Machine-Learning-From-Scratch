import numpy as np
from Cost import Cost
import DataReader


class GradientDescent():
    def LinerGradientDescent(data_x, data_y, epsilon=0.01, alpha=0.001):
        count = 0
        theta = [0, 0]
        diff = 0
        while True:
            count += 1
            for index in range(len(data_x)):
                diff = theta[0] + theta[1] * data_x[index] - data_y[index]

                theta[0] -= alpha * diff
                theta[1] -= alpha * diff * data_x[index]

                predict_y = theta[0] + theta[1] * data_x

                predictCost0 = Cost.SquaredErrors(data_x, predict_y)

                if count % 100 == 0:
                    print(count, predictCost0)

        if predictCost0 - predictCost1 < epsilon:
            return theta, predictCost
        predictCost1 = predictCost0

    def G1radientDescent(data_x, data_y, epsilon=0.001, alpha=0.001):
        count = 0
        theta = [0, 0]
        diff = 0
        while True:
            count += 1
            for index in range(len(data_x)):
                diff = 0
                for i in range(order):
                    if data_x[index] != 0 or i != 0:
                        diff += theta[i] * data_x[index]**i - data_y[index]
                for i in range(order):
                    theta[i] -= alpha * diff * data_x[index]
                predict_y = np.zeros(data_x.shape)
                for i in range(order):
                    predict_y += theta[i] * data_x**i
                predictCost0 = Cost.SquaredErrors(data_x, predict_y)
                if count % 100 == 0:
                    print(count, predictCost)
        if predictCost0 - predictCost1 < epsilon:
            return theta, predictCost
        predictCost1 = predictCost0


data_x, data_y = DataReader.PrebuiltData.MyGivenData()
theta, cost = GradientDescent.LinerGradientDescent(data_x, data_y)
print(theta, cost)