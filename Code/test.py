import numpy as np
from Cost import Cost
import DataReader


class GradientDescent():
    def GradientDescent(data_x, data_y, order, alpha=0.001, error=0.001):
        count = 0
        order += 1
        for i in range(order):
            theta.append(np.random.randint(100))
            diff.append(0)
        dataCost = Cost.SquaredErrors(data_x, data_y)
        predictCost = 0x7fffffff
        while predictCost - dataCost > error:
            count += 1
            for index in range(len(data_x)):
                for i in range(order):
                    for j in range(order):
                        if data_x[index] != 0 or i != 0:
                            diff[i] += (theta[j] * data_x[index]**i)
                for i in range(order):
                    theta[i] -= alpha * diff[i]
                predict_y = np.array(theta0 + theta1 * data_x)
                predictCost = Cost.SquaredErrors(data_x, predict_y)
                if count % 100 == 0:
                    print(count, predictCost)
        return theta, predictCost


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


data_x, data_y = DataReader.PrebuiltData.MyGivenData()
k, b = LinerLeastSquaresMethod(data_x, data_y)
predict_y = np.array(data_x * k + b)
print(k, b)
print(Cost.SquaredErrors(data_y, predict_y))
'''theta, cost = GradientDescent.LinerGradientDescent(data_x, data_y)
print(theta, cost)'''