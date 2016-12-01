import sys
import os
# sys.path.append(os.path.abspath("../src/"))
import SalientObject
import time
import glob
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm, grid_search
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib


def find_feature_vector(new_map, faceInfo, x_start, grid_size = (9, 9), block_size = (3, 3), nbins = 20, full_composition=True):
    """
        Extract photographic composition vector for an image

        """

        n_xcells, n_ycells = grid_size
        n_bxcells, n_bycells = block_size

        # cell size for a image
        img_height, img_width = new_map.shape
        x_step = 1.0*img_height/n_xcells
        y_step = 1.0*img_width/n_ycells

        # block steps
        bx_step = x_step*n_bxcells
        by_step = y_step*n_bycells

        n_xblocks = n_xcells - n_bxcells + 1
        n_yblocks = n_ycells - n_bycells + 1

        n_dims = nbins*n_xblocks*n_yblocks
        feature_vector = np.zeros(n_dims)

        x_pos = 0.0
        y_pos = 0.0
        count = 0
        for i in range(n_xblocks):
            # reset y_pos to 0
            y_pos = 0.0
            for j in range(n_yblocks):
                x_min = int(x_pos)
                y_min = int(y_pos)
                x_max = int(x_pos + bx_step)
                y_max = int(y_pos + by_step)

                # extract histogram for this block
                saliency_block = new_map[x_min:x_max, y_min:y_max]
                hist, bin_edges = np.histogram(saliency_block, nbins, (0,1))

                feature_vector[count:count+nbins] = hist
                count += nbins

                # increment by one cell step
                y_pos = (j+1)*y_step

            # increment x_step by one cell
            x_pos = (i+1)*x_step

        # print feature_vector.shape
        cv2.normalize(feature_vector, feature_vector, 0, 1, cv2.NORM_MINMAX)
        return feature_vector

def modify_map(new_map, faces, x_start):
    seg_map2 = np.copy(new_map)
    for (x, y, w, h) in faces:
        # x - horizontal position
        # y - vertical position
        # w - width
        # h - height
        for i in range(w):
            for j in range(h):
                seg_map2[y+j][x_start+x+i] = 1.0

    return seg_map2

def findSuggestion(image_src, dump_path, db):

    my_object = SalientObject.SalientObjectDetection(image_src)
    # my_object.plot_maps(dump_path)
    # my_object.process_segments(dump_path)
    
    comp_map, feature_v, face_v, view_c = my_object.get_photograph_composition()

    new_map = my_object.getNewSaliencyMap()
    faceInfo = my_object.getFaceInfo()

    dump_scalar = db + "/scalar.pkl"
    dump_pca = db + "/pca.pkl"
    dump_svm = db + "/svm.pkl"

    regressor = joblib.load(dump_svm)
    scalar = joblib.load(dump_scalar)
    pca = joblib.load(dump_pca)

    # form feature
    X = []
    X.extend(feature_v)
    X.extend(view_c)
    # X.extend(geo)

    X = np.asarray(X)
    print X.shape
    X = np.reshape(X, (1, -1))
    print X.shape

    X = scalar.transform(X)
    X = preprocessing.normalize(X, norm='l2')
    X = pca.transform(X)
    pred = regressor.predict_proba(X)

    print pred

    img_height, img_width = new_map.shape
    # find bounding box for faces
    min_x, min_y = 10000, 10000
    max_x, max_y = 0, 0, 0, 0
    for (x, y, w, h) in faceInfo:
        # x - horizontal position
        # y - vertical position
        # w - width
        # h - height
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y

        if max_x < x+w:
            max_x = x+w
        if max_y < y+h:
            max_y = y+h

    print min_x, min_y, max_x, max_y
    box_h = max_y - min_y
    box_w = max_x - min_x

    # delete faces first
    for (x, y, w, h) in faceInfo:
            # x - horizontal position
            # y - vertical position
            # w - width
            # h - height
            for i in range(w):
                for j in range(h):
                    new_map[y+j][x+i] = 0.0

    x_start = 0
    y_start = min_y
    x_step = int(img_width/20)

    # iterate in grid
    n_step = int(img_width - box_w)/x_step
    for ii in range(n_step):

        modified_map = modify_map(new_map, faceInfo, x_start)
        fv = find_feature_vector(modified_map)

        # form feature
        X = []
        X.extend(fv)
        X.extend(view_c)
        # X.extend(geo)

        X = np.asarray(X)
        # print X.shape
        X = np.reshape(X, (1, -1))
        # print X.shape

        X = scalar.transform(X)
        X = preprocessing.normalize(X, norm='l2')
        X = pca.transform(X)
        pred = regressor.predict_proba(X)

        print pred

        x_start += x_step


def testing(w_dir, image_dir, dump_path, model_path, file_list='image.list'):
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    image_list = w_dir + '/' + file_list
    fp_image_list = open(image_list, 'r')

    f_geo = w_dir + "/geo.info"
    # geo_info = np.loadtxt(f_geo)

    i = 0

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\r\n")
        infile = image_dir + image_name

        print infile

        timer = time.time()
        findSuggestion(infile, dump_path, model_path)
        
        print "Total run time = ", time.time() - timer

        i += 1

# w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/"
# db_src = w_dir + "ImageDB_640/"
# dump_path = w_dir + "dump_Image_640/"

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "Usage : dataset_path"
        sys.exit(0)

    w_dir = sys.argv[1]

    m_path = "/mnt/windows/DataSet-YsR/merlion/"
    # w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/"
    db_src = w_dir + "/ImageDB/"
    dump_path = w_dir + "/dump_ImageDB/"

    testing(w_dir, db_src, dump_path, file_list='image.list', model_path=m_path)

