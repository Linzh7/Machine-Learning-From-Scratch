import numpy as np
import math
from Cost import Cost


class GradientDescent():
    def LinerGradientDescent(alpha=0.001, data_x, data_y):
        count = 0
        theta = [0]
        '''
        for i in np.arange(thetaNum):
            theta.append(0)
        '''
        while ():
            count += 1
            for index in np.arange(len(data_x)):
                diff = (theta[0] + theta[1] * data_x) - data_y
                for i in np.arange(len(theta)):
                    theta[i] -= alpha * diff * x[index][i]
                predict_y = np.array(map((lambda x: theta0 + theta1 * x, data_x))
                cost=Cost.LinerRegressionCost(data_x, predict_y)
                print(cost)
                