import numpy as np


def hr_score(true, pred, k=4):
    topk_ = pred.argsort(axis=1)[:, ::-1][:, :k]
    acc = np.max(topk_ == true, axis=1)
    acc = np.mean(acc)

    return acc


def ndcg_score(true, pred, k=4):
    """Compute ndcg score given true values and predicted values"""
    topk_ = pred.argsort(axis=1)[:, ::-1][:, :k]
    # Discount array
    discount = np.log2(np.arange(2, topk_.shape[1] + 2)).reshape(1, -1)

    # Get relevance, i.e. check if any items in pred matches any items in true
    rel = np.array([np.isin(topk_[i], true[i]) for i in range(true.shape[0])])

    # Get ideal relevance
    irel = np.zeros(topk_.shape)
    irel[:, : true.shape[1]] = 1

    # Compute dcg, idcg
    dcg = np.divide(rel, discount).sum(axis=1)
    idcg = np.divide(irel, discount).sum(axis=1)

    return (dcg / idcg).mean()
