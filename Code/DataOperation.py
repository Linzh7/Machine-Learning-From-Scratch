def FillMissingData(self, sets):
    for index in range(sets.shape[0]):
        if sets[index] == -1:
            sets[index] = sets.mean()
    return sets


def LablePressing(self, data):
    dic = {}
    for lable in data:
        if lable in dic:
            dic[lable] += 1
        else:
            dic[lable] = 1
    mapping = []
    count = 1
    for k, v in dic.items():
        mapping.append([count, k])
    return mapping


def Normalization(self, sets):
    for i in range(len(sets[0])):
        sets[:, i] = (sets[:, i] - sets[:, i].min()) / sets[:, i].max()
    return sets