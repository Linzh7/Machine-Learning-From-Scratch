import numpy as np


class PrebuiltData():
    def Fx(x, a=0, b=3, c=3):
        return a * x**2 + b * x + c

    def MyData(rangeStart=1,
               rangeEnd=100,
               rangeNum=100,
               randomFloat=1,
               a=0,
               b=3,
               c=3):
        data_x = np.linspace(rangeStart, rangeEnd, rangeNum)
        random = np.random.randn(len(data_x))
        data_y = np.array(
            PrebuiltData.Fx(data_x, a, b, c) + randomFloat * random)
        return data_x, data_y

    def MyGivenData():
        data_x = np.array(range(1, 100))
        data_y = [
            3.03, 8.98, 9.76, 17.77, 19.36, 18.02, 21.67, 24.84, 30.06, 31.09,
            33.14, 39.66, 43.24, 43.01, 49.08, 48.67, 51.22, 58.73, 62.35,
            63.37, 66.21, 68.36, 72.58, 77.73, 79.30, 82.41, 84.59, 87.65,
            92.71, 94.14, 98.96, 96.92, 100.49, 103.69, 107.47, 112.30, 112.25,
            119.65, 120.27, 122.58, 126.75, 126.56, 131.66, 135.19, 137.22,
            140.52, 144.36, 148.66, 151.89, 151.59, 154.52, 156.10, 160.59,
            167.87, 169.54, 169.42, 175.30, 176.61, 181.15, 183.19, 187.32,
            188.07, 194.08, 192.35, 195.48, 199.27, 202.93, 208.60, 209.41,
            210.97, 216.89, 216.22, 222.08, 226.80, 225.78, 230.25, 231.79,
            234.64, 242.19, 241.19, 243.91, 246.65, 254.63, 254.22, 260.18,
            261.71, 264.59, 267.76, 268.21, 274.39, 275.98, 280.82, 283.98,
            282.57, 285.02, 288.28, 296.47, 294.98, 299.00
        ]
        return np.array(data_x), np.array(data_y)

    def MyMultivariateGivenData(a=100, b=200, d=2, e=3, f=1):
        data_x = np.append([range(1, a)],
                           [range(2, b, b // a)]).reshape(2, a - 1)
        data_y = np.array(data_x[0] * d + data_x[1] * e +
                          f * np.random.normal())
        return data_x, data_y

    #3a+5b+normal()
    def MyMultivariateGivenData():
        data_x = np.append([range(1, 100)], [range(2, 200, 2)]).reshape(2, 99)
        data_y = np.array([
            11.60, 24.60, 37.60, 50.60, 63.60, 76.60, 89.60, 102.60, 115.60,
            128.60, 141.60, 154.60, 167.60, 180.60, 193.60, 206.60, 219.60,
            232.60, 245.60, 258.60, 271.60, 284.60, 297.60, 310.60, 323.60,
            336.60, 349.60, 362.60, 375.60, 388.60, 401.60, 414.60, 427.60,
            440.60, 453.60, 466.60, 479.60, 492.60, 505.60, 518.60, 531.60,
            544.60, 557.60, 570.60, 583.60, 596.60, 609.60, 622.60, 635.60,
            648.60, 661.60, 674.60, 687.60, 700.60, 713.60, 726.60, 739.60,
            752.60, 765.60, 778.60, 791.60, 804.60, 817.60, 830.60, 843.60,
            856.60, 869.60, 882.60, 895.60, 908.60, 921.60, 934.60, 947.60,
            960.60, 973.60, 986.60, 999.60, 1012.60, 1025.60, 1038.60, 1051.60,
            1064.60, 1077.60, 1090.60, 1103.60, 1116.60, 1129.60, 1142.60,
            1155.60, 1168.60, 1181.60, 1194.60, 1207.60, 1220.60, 1233.60,
            1246.60, 1259.60, 1272.60, 1285.60
        ])
        return data_x, data_y
