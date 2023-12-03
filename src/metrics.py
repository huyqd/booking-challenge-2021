import numpy as np


def topk(preds, target, k=4):
    topk_ = preds.argsort(axis=1)[:, ::-1][:, :k]
    acc = np.max(topk_ == target, axis=1)
    acc = np.mean(acc)

    return acc
