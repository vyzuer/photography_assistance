import Image
import time

import numpy as np
import pylab as pl

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
from sklearn.preprocessing import scale, normalize

def k_means(dump_path, file_name, n_clusters=10):
    # Obtain data from file.
    #feature_file = 'feature.list'
    data = np.loadtxt(file_name)
    print data.shape
    
    # use only RGB
    X = data[:, 136:232]
    # X = scale(data)
    # X = normalize(X, norm='l2')
    
    # Compute clustering with Means
    print "started k-means..."
    # k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_iter=10000000, tol=0.0000001, n_jobs=-1)
    k_means = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_iter=100000, tol=0.000001, max_no_improvement=None)

    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0
    print "k-means done."
    k_means_labels = k_means.labels_
    #print k_means_labels
    label_file = dump_path + "labels.list"
    fp = open(label_file, 'w')
    for i in k_means_labels:
        fp.write("%d\n" % i)
    fp.close()

    num_cluster_file = dump_path + "_num_clusters.info"
    fp = open(num_cluster_file, 'w')
    fp.write("%d" % n_clusters)
    fp.close()


    k_means_cluster_centers = k_means.cluster_centers_
    
    centre_file = dump_path + "_centers.info"
    np.savetxt(centre_file, k_means_cluster_centers)

    score = 0

    # print "evaluating performance..."
    # score = metrics.silhouette_score(X, k_means_labels, metric='euclidean', sample_size=20000)
    # print "evaluation done."
    # score = metrics.silhouette_samples(X, k_means_labels, metric='euclidean', sample_size=1000)
    # score = np.sum(score)/len(score)

    return score

