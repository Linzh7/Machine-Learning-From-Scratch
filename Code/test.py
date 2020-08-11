import numpy as np
from Cost import Cost
import DataReader


class GradientDescent():
    def GradientDescent(data_x, data_y, order, error=1, alpha=0.001):
        count = 0
        order += 1
        theta = []
        diff = []
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
                predict_y = np.zeros(data_x.shape)
                for i in range(order):
                    predict_y += theta[i] * data_x**i
                predictCost = Cost.SquaredErrors(data_x, predict_y)
                if count % 100 == 0:
                    print(count, predictCost)
        return theta, predictCost


data_x, data_y = DataReader.PrebuiltData.MyGivenData()
theta, cost = GradientDescent.GradientDescent(data_x, data_y, 1)
print(theta, cost)