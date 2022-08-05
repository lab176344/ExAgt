from src.evaluation.eval_0 import eval_0
from sklearn.cluster import KMeans

def kmeans(data,labels,num_unlabeled_classes):
    kmeans = KMeans(n_clusters=num_unlabeled_classes,n_init=20).fit(data)         
    y = kmeans.labels_
    eval_acc = eval_0()
    acc = eval_acc.param_calc(labels.astype(int), y.astype(int))
    print('K means acc {:.4f}'.format(acc))
    return acc