import Image
import numpy as np
from numpy import linalg
from matplotlib import pyplot
import sys
from sklearn.decomposition import PCA, RandomizedPCA, KernelPCA
from sklearn.decomposition import ProjectedGradientNMF
from sklearn import mixture
import itertools
from sklearn import preprocessing
from scipy import linalg
import pylab as pl
import matplotlib as mpl
from sklearn import metrics
import os
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
from sklearn.preprocessing import scale, normalize
from sklearn.externals import joblib
    

# DIM_X, DIM_Y = 40, 60 # 
# grid_height, grid_width = 4, 6
# block_height, block_width = 10, 10

DIM_X, DIM_Y = 24, 32 
grid_height, grid_width = 7, 7
block_height, block_width = 5, 5

RULE_X, RULE_Y = 6, 8

def find_eigen_rules(data, w_dir, model_path):
    pca_dump_dir = model_path + "pca/"

    pca_dump = pca_dump_dir + "pca.pkl"

    find_popular_rules(data, w_dir, pca_dump)


def find_base_rules(data, w_dir, model_path):
    nmf_dump_dir = model_path + "nmf/"

    nmf_dump = nmf_dump_dir + "nmf24.pkl"

    nmf = perform_nmf(data, w_dir, nmf_dump)


def main(w_dir, model_path):

    comp_list = w_dir + "feature_dumps/comp_map.list"

    print "Reading composition vectors..."
    X = np.loadtxt(comp_list)
    print X.shape

    find_eigen_rules(X, w_dir, model_path)

    print "done."

def scale_data(X):
    # _scalar = preprocessing.StandardScaler()
    _scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))

    X = _scalar.fit_transform(X)
    # X = preprocessing.normalize(X, norm='l2')

    return X


def ap(X):    
    ##############################################################################
    # Compute Affinity Propagation
    af = AffinityPropagation(damping=0.9, convergence_iter=50, max_iter=1000).fit(X)
    print af.cluster_centers_indices_
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    
    n_clusters_ = len(cluster_centers_indices)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

    visualize_cluster_rules(X, cluster_centers_indices, 50)


def dbscan(X):
    db = DBSCAN(eps=0.95, min_samples=5).fit(X)
    core_samples = db.core_sample_indices_
    components = db.components_
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print('Estimated number of clusters: %d' % n_clusters)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    visualize_cluster_rules_1(components, 10)
    

def kmeans(X):
    k_means = KMeans(init='k-means++', n_clusters=10, n_init=10, max_iter=10000, tol=0.00001).fit(X)
    labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_

    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    visualize_cluster_rules_1(k_means_cluster_centers, 10)


def perform_clustering(X):

    # ap(X)

    dbscan(X)

    # kmeans(X)

def visualize_cluster_rules_1(rules, num_comp):

    for i in range(num_comp):
        base_rule = rules[i]
        base_rule = np.reshape(base_rule, (DIM_X, DIM_Y)) 
        pyplot.subplot(2,10,i+1)
        pyplot.axis('off')
        pyplot.imshow(base_rule, origin="upper")

    pyplot.show()


def visualize_cluster_rules(X, rules, num_comp):

    for i in range(num_comp):
        base_rule = X[rules[i]]
        base_rule = np.reshape(base_rule, (DIM_X, DIM_Y)) 
        pyplot.subplot(5,10,i+1)
        pyplot.axis('off')
        pyplot.imshow(base_rule, origin="upper")

    pyplot.show()


def test_gmm(X):

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 20)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    
    bic = np.array(bic)
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    clf = best_gmm
    bars = []
    
    # Plot the BIC scores
    spl = pl.subplot(1, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(pl.bar(xpos, bic[i * len(n_components_range):
                                     (i + 1) * len(n_components_range)],
                           width=.2, color=color))
    pl.xticks(n_components_range)
    pl.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    pl.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    pl.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    pl.show()


def perform_gmm(X):

    # test_gmm(X)

    print "Performing GMM..."
    gmm = mixture.GMM(n_components=20, covariance_type='diag')
    gmm.fit(X)
    rules = gmm.means_
    print "done."

    # visualize GMM rules
    rules = project_data(rules)
    visualize_gmm_rules(rules, 20)

def plot_variance(data, f_name):
    #evaluate the cumulative
    cumulative = np.cumsum(data)
    x = np.arange(1,40)
    # plot the cumulative function
    pyplot.plot(x, cumulative[0:50], c='blue')
    pyplot.savefig(f_name)
    pyplot.close()

def find_popular_rules(X, w_dir, pca_dump):

    n_clusters = 100

    pca = joblib.load(pca_dump)

    data = pca.transform(X)

    k_means = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_iter=2000, tol=0.0, max_no_improvement=50)

    k_means.fit(data)

    e_rules = k_means.cluster_centers_

    e_rules = pca.inverse_transform(e_rules)

    visualize_popular_rules(e_rules, w_dir, n_clusters)


def scale_data_2(X):
    # _scalar = preprocessing.StandardScaler()
    _scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))

    X = _scalar.fit_transform(X)
    X = preprocessing.normalize(X, norm='l2')

    return X

def project_data(X):
    height, width = X.shape
    for i in range(height):
        X[i] = map_feature_to_composition(X[i])

    return X

def perform_nmf(X, w_dir, model_path):

    n_com = 24
    model = joblib.load(model_path)
    nmf_components = model.transform(X) 

    # visualize Base Rules
    visualize_base_rules(nmf_components, n_com, w_dir)

    return model

def map_feature_to_composition(vector):

    composition_height = grid_height*block_height
    composition_width = grid_width*block_width

    composition_map = np.zeros(shape=(composition_height, composition_width))
    # map back the feature vector to image composition
    cnt = 0
    for i in range(grid_height):
        for j in range(grid_width):
            for k in range(block_height):
                for l in range(block_width):
                    ii = i*block_height + k
                    jj = j*block_width + l
                    composition_map[ii][jj] = vector[cnt]
                    cnt +=1
    
    vec_map = np.reshape(composition_map, -1, 1)

    return vec_map

def read_composition(w_dir, dump_dir):
    composition_list = dump_dir + "comp_map.fv"
    f_score = w_dir + "aesthetic.scores"

    data = np.loadtxt(composition_list)
    a_score = np.loadtxt(f_score)
    x = len(a_score)

    X = []
    Y = []

    for i in range(x):
        if a_score[i] > 0.60:
            X.append(data[i])
        elif a_score[i] < 0.30:
            Y.append(data[i])

    X = np.asarray(X)
    Y = np.asarray(Y)

    return data, X, Y

def visualize_gmm_rules(V, num_comp):
    # visualize GMM centers
    for i in range(num_comp):
        base_rule = V[i]
        base_rule = np.reshape(base_rule, (DIM_X, DIM_Y)) 
        pyplot.subplot(4,5,i+1)
        pyplot.axis('off')
        pyplot.imshow(base_rule, origin="upper")

    pyplot.show()

    return


def visualize_base_rules(V, num_comp, f_name):
    dump_dir = w_dir + "/base_rules_map/"
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    image_list = w_dir + "image.list"
    fp_image_list = open(image_list, 'r')

    i = 0
    for image_name in fp_image_list:
        img_name = image_name.rstrip("\n")
        f_name = dump_dir + img_name

        base_rule = V[i]
        base_rule = np.reshape(base_rule, (4, 6)) 
        pyplot.axis('off')
        pyplot.matshow(base_rule)
        pyplot.savefig(f_name)
        pyplot.close()

        i += 1

    return

def visualize_popular_rules(V, w_dir, num_rules):
    f_name = w_dir + "popular_compositoin.png"
    for i in range(num_rules):
        eigen_rule = V[i]
        eigen_rule = np.reshape(eigen_rule, (DIM_X, DIM_Y))
        pyplot.subplot(10,10,i+1)
        pyplot.axis('off')
        eigen_rule = np.absolute(eigen_rule)
        pyplot.imshow(eigen_rule, origin="upper")

    pyplot.savefig(f_name)
    pyplot.close()

    return


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage : dataset_path model_path"
        sys.exit(0)

    w_dir = sys.argv[1]
    model_path = sys.argv[2]

    main(w_dir, model_path)

