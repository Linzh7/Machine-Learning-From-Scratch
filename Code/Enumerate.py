import numpy as np
import Cost


class Enumerate():
    def LinerRegressionEnumerate(data_x, data_y, maxRange=100, stepSize=1):
        cost = 0x7fffffff
        costSave = 0x7fffffff
        kSave = 0
        bSave = 0
        for k in range(0, maxRange, stepSize):
            for b in range(0, maxRange, stepSize):
                predict_y = list(map(lambda x: x * k + b, data_x))
                cost = Cost.LinerRegressionCost(data_y, predict_y, k, b)
                if cost < costSave:
                    kSave = k
                    bSave = b
                    costSave = cost
                    #print(k, b, cost)
        return kSave, bSave, costSave