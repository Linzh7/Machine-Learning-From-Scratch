import numpy as np
from Cost import Cost
import DataReader


class GradientDescent():
    def LinerGradientDescent(data_x, data_y, epsilon=0.01, alpha=0.001):
        count = 0
        theta = [0, 0]
        while True:
            count += 1
            gradient = [0, 0]
            for index in range(len(data_x)):
                diff = (theta[0] + theta[1] * data_x[index]) - data_y[index]
                gradient[0] += (2 / len(data_x)) * diff
                gradient[1] += (2 / len(data_x)) * data_x[index] * diff

            theta[0] -= alpha * gradient[0]
            theta[1] -= alpha * gradient[1]

            predict_y = theta[0] + theta[1] * data_x

            predictCost0 = Cost.SquaredErrors(data_x, predict_y)

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
theta, cost = GradientDescent.step_gradient(data_x, data_y)
print(theta, cost)