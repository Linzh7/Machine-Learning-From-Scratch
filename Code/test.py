import numpy as np
import math
import DataReader
from Cost import Cost


class GradientDescent():
    def BatchGradientDescent(data_x,
                             data_y,
                             variableNum,
                             epsilon=0.01,
                             learningRate=0.001):
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


#3a + 0.5b + normal()
def MyMultivariateGivenData():
    data_x = []
    for i in range(1, 100):
        data_x.append([i * 1.0, 20.0 * i])
    data_y = np.array([
        11.60, 24.60, 37.60, 50.60, 63.60, 76.60, 89.60, 102.60, 115.60,
        128.60, 141.60, 154.60, 167.60, 180.60, 193.60, 206.60, 219.60, 232.60,
        245.60, 258.60, 271.60, 284.60, 297.60, 310.60, 323.60, 336.60, 349.60,
        362.60, 375.60, 388.60, 401.60, 414.60, 427.60, 440.60, 453.60, 466.60,
        479.60, 492.60, 505.60, 518.60, 531.60, 544.60, 557.60, 570.60, 583.60,
        596.60, 609.60, 622.60, 635.60, 648.60, 661.60, 674.60, 687.60, 700.60,
        713.60, 726.60, 739.60, 752.60, 765.60, 778.60, 791.60, 804.60, 817.60,
        830.60, 843.60, 856.60, 869.60, 882.60, 895.60, 908.60, 921.60, 934.60,
        947.60, 960.60, 973.60, 986.60, 999.60, 1012.60, 1025.60, 1038.60,
        1051.60, 1064.60, 1077.60, 1090.60, 1103.60, 1116.60, 1129.60, 1142.60,
        1155.60, 1168.60, 1181.60, 1194.60, 1207.60, 1220.60, 1233.60, 1246.60,
        1259.60, 1272.60, 1285.60
    ])
    return np.array(data_x), data_y


def Normalization(sets):
    for i in range(len(sets[0])):
        sets[:, i] = sets[:, i] / sets[:, i].max()
    return sets


data_x, data_y = MyMultivariateGivenData()
data_x = Normalization(data_x)
theta, cost = GradientDescent.BatchGradientDescent(data_x, data_y, 2)
print(theta, cost)