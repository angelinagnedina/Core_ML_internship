#!/usr/bin/env python
# coding: utf-8




import numpy as np


def precisionk(prediction, truth, k: int) -> float:
    return np.sum(np.multiply(prediction, truth), axis = 0)/k


def apk(prediction, truth, K: int, threshold: float) -> float:
    prediction = list(map(lambda x: int(x >= threshold), prediction))
    truth = list(map(lambda x: int(x >= threshold), truth))
    new_arr = [[prediction[i], truth[i]] 
                        for i in range(len(truth))]
    new_arr.sort(reverse=True)
    new_arr = np.array(new_arr)
    predictions, truth = new_arr[:,0][:K], new_arr[:,1][:K]
    res = 0.0
    for k in range(1, K + 1):
        res += truth[k - 1]*precisionk(predictions[:k], truth[:k], k)
    return res / K
 

def ndcgk(prediction, truth, K: int) -> float:
    new_arr = [[prediction[i], truth[i]]
                        for i in range(len(truth))]
    new_arr.sort(reverse=True)
    new_arr = np.array(new_arr)
    pred = new_arr[:,1][:K]
    truth = sorted(truth, reverse=True)
    res = 0.0
    denom = 0.0
    for k in range(1, K + 1):
        res += (2**(pred[k - 1]) - 1)/np.log2(k + 1)
        denom += (2**(truth[k - 1]) - 1)/np.log2(k + 1)
    return res/denom