from collections import Counter
from utils import recall_at_k, ndcg_k
import numpy as np

f = open('../data/Beauty.txt')

counter = Counter()
answers = []
for l in f:
    item_ids = l.split(" ", 1)[1].split() # index of the item_ids is 1
    counter.update(item_ids) # only use training data that the rest of the sequential algorithms use hence the idx stops at -2
    answers.append((item_ids[-1:])) # get the last item of the sequence for testing purposes 

print(counter.most_common(20))
ids, _ = zip(*counter.most_common(20))
ids = np.repeat(np.array([ids]), len(answers), axis=0)

answers = np.array(answers, dtype='object')

recall, ndcg = [], []
for k in [5, 10, 20]:
    recall.append(recall_at_k(answers, ids, k))
    ndcg.append(ndcg_k(answers, ids, k))
post_fix = {
    "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
    "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
    "HIT@20": '{:.4f}'.format(recall[2]), "NDCG@20": '{:.4f}'.format(ndcg[2])
}
print(post_fix)
