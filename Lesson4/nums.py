import sklearn
from sklearn.datasets import load_digits
import numpy as np
import time
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)

y = digits.target

# k = len(np.unique(y))
k = 10
samples, features = data.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric="euclidean")))
 

clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)

# digits = load_digits()
#
# data = scale(digits.data)
#
# n_samples, n_features = data.shape
# n_digits = len(np.unique(digits.target))
# labels = digits.target
#
# sample_size = 300
#
# print("n_digits: %d, \t n_samples %d, \t n_features %d"
#       % (n_digits, n_samples, n_features))
#
# print(82 * '_')
# print('init\ttime\tinertia\thomo\tcompl\tv=meas\tART\tAMI\tsilhouette')
#
#
# def bench_k_means(estimator, name, data):
#     t0 = time()
#     estimator.fit(data)
