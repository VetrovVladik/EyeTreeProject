import numpy as np
from collections import Counter
from collections import OrderedDict


def CalcIC(x=None, y=None, method=None):
    if method is None:
        method = Gini
    left = []
    right = []
    infoCriterion = 0
    minBoundary = 0
    minInfo = 1
    boundary = 0
    xy = np.transpose(np.array([x, y]))
    xy = xy[xy[:, 0].argsort()]
    if (len(set(x)) <= 2):
        boundary = np.mean(np.array(set(x)))
        initVal = xy[:, 0][0]
        for ind, val in enumerate(xy[:, 0]):
            if val != initVal:
                left = xy[:, 0][0:ind - 1]
                right = xy[:, 0][ind:]
                break
        infoCriterion = method(y) - method(left) * len(left) / len(y) - method(right) * len(right) / len(y)
        minBoundary = boundary
    else:
        for ind, val in enumerate(xy[:, 0]):
            if (ind - 1) >= 0:
                boundary = (val + xy[:, 0][ind - 1]) / 2
            else:
                boundary = val
            left = xy[:, 0][0:ind - 1]
            right = xy[:, 0][ind:]
            infoCriterion = method(y) - method(left) * len(left) / len(y) - method(right) * len(right) / len(y)

            if (infoCriterion < minInfo):
                minInfo = infoCriterion
                minBoundary = boundary

    return [minInfo, minBoundary]


def DoSplit(x, y, boundary):
    indicies = [i for i in range(len(y))]
    xyi = np.transpose(np.array([x, y, indicies]))
    leftIndicies = []
    rightIndicies = []
    xyi = xyi[xyi[:, 0].argsort()]
    for ind, val in enumerate(xyi[:, 0]):
        if val <= boundary:
            left.append(xyi[:, 2][ind])
        else:
            left.append(xyi[:, 2][ind])


return dict(["left", "right"], [leftIndicies, rightIndicies])


def Gini(y):
    cnt = Counter()
    dct = cnt(y)
    sm = np.sum(y)
    acc = 0;
    for pair in dct.most_common():
        acc += (pair[1] / sm) ** 2
    return 1 - acc


def MSE(y):
    pred = np.mean(y)
    acc = 0
    for num in y:
        acc += (num - pred) ** 2
    return acc / len(y)


def Entropy(y):
    cnt = Counter()
    dct = cnt(y)
    sm = np.sum(y)
    acc = 0;
    print("for Git commit")
    for pair in dct.most_common():
        acc += np.log((pair[1] / sm)) * (pair[1] / sm)
    return -acc


class EyeBinaryTreeNode:
    def __init__(self, X=None, y=None, max_depth=10, min_sample_leaf=2, level, eyes=1, bof=3, criterion):
        self._level = level
        self._left = None  # It's either EyeTreeNode or None
        self._right = None  # It's either EyeTreeNode or None
        self._factor = None  # Should be number
        self._boundary = None  # Boundary
        self._prediction = None  # Prediction number or None
        self._X = X  # Array of factors (truncated)
        self._y = y  # Vector of y (truncated)
        self._min_sample_leaf = min_sample_leaf
        self._prediction = []
        self.Construct()

    def AddPrediction(self, pred):
        self._prediction.add(pred)

    def Construct(self):
        binaryType = type(EyeBinaryTreeNode())
        if len(self._y[:, 0]) >= self._min_sample_leaf:
            numFactors = factors.shape[1]
            criteriaDict = dict()
            for num in range(numFactors):
                criteriaDict[num] = CalcIC(self._X[:, num], self._y[:, 0], criterion)
            orDict = OrderedDict(sorted(criteriaDict.items(), key=lambda v: v[0], reverse=True))
            if eyes >= 1:
                t = 2
            else:
                XN = self._X
                yN = self._y
                opt = list(orDict.keys())[0]
                self._factor = opt
                splitBoundary = orderedDict.get(opt)[1]
                self._boundary = splitBoundary
                splitIndicies = DoSplit(XN[:, opt], yN, splitBoundary)

                leftX = np.array([list(XN[i, :]) for i in splitIndicies['left']])
                leftY = np.array([list(yN[i, :]) for i in splitIndicies['left']])

                rightX = np.array([list(XN[i, :]) for i in splitIndicies['right']])
                rightY = np.array([list(yN[i, :]) for i in splitIndicies['right']])

                if (len(leftY) >= self._min_sample_leaf) & & self._level + 1 < max_depth:
                    self._left = EyeBinaryTreeNode(X=leftX, y=leftY, level=self._level + 1)
                else:
                    self._left = np.mean(leftY)

                if len(rightY) >= self._min_sample_leaf & & self._level + 1 < max_depth:
                    self._right = EyeBinaryTreeNode(X=rightX, y=rightY, level=self._level + 1)
                else:
                    self._right = np.mean(rightY)

    def Score(self, X=None):
        scored = []
        if X is None:
            print("No data!")
            return
        _X = np.transpose(np.array(X))
        for x in _X:
            if self

        return score


class EyeDecisionTree:
    def __init__(self, criterion=None, max_depth=10, min_sample_leaf=2, eyes=0, bof=3):
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_sample_leaf = min_sample_leaf
        self._eyes = eyes
        self._bof = bof

    def Fit(self, X=None, y=None, criterion="gini"):
        if len(X) == len(y):
            _X = np.transpose(np.array(X))
            _y = np.transpose(np.array(X))
        else:
            print("Something is wrong with input.")
            return
        self._tree = EyeBinaryTreeNode(_X, _y, self._max_depth, self._min_sample_leaf, 0, self._eyes, self._bof,
                                       self_.criterion)

    def Predict(self, X=None):
        if X is None:
            print("No data!")
            return
        _X = np.transpose(np.array(X))
        y = []
        for point in _X:
            if self._tree.
        return score
