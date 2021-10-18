import numpy as np

# Global variables
_model = None
_trainRatings = None
_testRatings = None
_numItems = None
_numFactors = None
_testNegatives = None

def evaluate_model(model, trainRatings, testRatings, testNegatives ,num_items, num_factors):
    global _model
    global _trainRatings
    global _testRatings
    global _testNegatives
    global _numItems
    global _numFactors

    _model = model
    _trainRatings = trainRatings
    _testRatings = testRatings
    _testNegatives = testNegatives
    _numItems = num_items
    _numFactors = num_factors

    # Single thread
    result = np.array([0.] * 8)
    for uid in list(_testRatings.keys()):
        re = simple_test_one_user(uid)
        result += re

    ret = result / len(list(_testRatings.keys()))
    ret = list(ret)
    return ret

def simple_test_one_user(x):
    u = x
    all_items = set(range(_numItems))
    # test_items = np.array(list(all_items - set(_trainRatings[u])))
    test_items = _testNegatives[u]
    test_items.extend(_testRatings[u])
    test_items = np.array(test_items)

    num_test = test_items.shape[0]

    # get item scores
    item_score = []
    test_users = np.full(len(test_items), u, dtype='int32')
    predictions, _, _ = _model.predict(
        [test_users, test_items, np.zeros((num_test, _numFactors)), np.zeros((num_test, _numFactors)),
         np.ones((num_test, _numFactors)), np.zeros((num_test, _numFactors)), np.zeros((num_test, _numFactors))],
        batch_size=num_test, verbose=0)

    for i in range(len(test_items)):
        item_score.append((test_items[i], predictions[i]))
    item_score = sorted(item_score, key=lambda x: x[1], reverse=True)
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in _testRatings[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])

    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    ap = average_precision(r)
    mrr = MRR(r)
    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10, ap, mrr])

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def MRR(r):
    rf = np.asarray(r).nonzero()[0]
    if rf.size == 0:
        return 0
    else:
        return 1/(rf[0]+1)
