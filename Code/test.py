import numpy as np
from Cost import Cost
import DataReader


class GradientDescent():
    def LinerGradientDescent(data_x, data_y, alpha=0.001, error=0.001):
        count = 0
        theta = [np.random.randint(100), np.random.randint(100)]
        diff = [0, 0]
        dataCost = Cost.SquaredErrors(data_x, data_y)
        predictCost = 0x7fffffff
        while predictCost - dataCost > error:
            count += 1
            for index in range(len(data_x)):
                for i in range(len(theta)):
                    for j in range(len(theta)):
                        if data_x[index] != 0 or i != 0:
                            diff[i] += (theta[j] * data_x[index]**i)
                for i in range(len(theta)):
                    theta[i] -= alpha * diff[i]
                predict_y = np.array(map(lambda x: theta0 + theta1 * x,
                                         data_x))
                predictCost = Cost.SquaredErrors(data_x, predict_y)
                if count % 100 == 0:
                    print(count, predictCost)
        return theta, predictCost


data_x, data_y = DataReader.PrebuiltData.MyData()
theta, cost = GradientDescent.LinerGradientDescent(data_x, data_y)
print(theta, cost)