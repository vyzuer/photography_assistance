import sys
import cv2
import os
# sys.path.append(os.path.abspath("../src/"))
import SalientObject
import time
import glob
import math
import numpy as np
from matplotlib import pyplot as plt
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

import cython_utils as cutils

init_pos = np.array([[1.0/6, 1.0/3], [1.0/6, 2.0/3], [1.0/3, 1.0/6], [1.0/3, 1.0/3], [1.0/3, 1.0/2], [1.0/3, 2.0/3], [1.0/3, 5.0/6], [1.0/2, 1.0/3], [1.0/2, 1.0/2], [1.0/2, 2.0/3], [2.0/3, 1.0/6], [2.0/3, 1.0/3], [2.0/3, 1.0/2], [2.0/3, 2.0/3], [2.0/3, 5.0/6], [5.0/6, 1.0/3], [5.0/6, 2.0/3]])
# init_pos = np.array([[1.0/3, 1.0/3], [1.0/3, 1.0/2], [1.0/3, 2.0/3], [1.0/2, 1.0/3], [1.0/2, 1.0/2], [1.0/2, 2.0/3], [2.0/3, 1.0/3], [2.0/3, 1.0/2], [2.0/3, 2.0/3]]])
# init_pos = np.array([[1.0/2, 1.0/2]])


def plot_test_image(dump_path, new_map):
    outfile = dump_path + "test.png"
    plt.imshow(new_map)
    plt.savefig(outfile)
    plt.close()


def find_feature_vector(new_map, grid_size = (9, 9), block_size = (3, 3), nbins = 20, full_composition=True):
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

def modify_map(new_map, faces, current_pos):
    
    seg_map2 = np.copy(new_map)
    faces_ = (faces*current_pos[2]/100).astype(np.int)
    # print faces_
    for (x, y, w, h) in faces_:
        # print x, y, w, h
        # x - horizontal position
        # y - vertical position
        # w - width
        # h - height
        
        for i in range(w):
            for j in range(h):
                try:
                    seg_map2[current_pos[0]+y+j][current_pos[1]+x+i] = 1.0
                except IndexError:
                    return seg_map2, False

    return seg_map2, True


def find_neighbors(current_pos, y_step, x_step, z_step):
    a = current_pos[0]-y_step, current_pos[1], current_pos[2]
    b = current_pos[0], current_pos[1]+x_step, current_pos[2]
    c = current_pos[0]+y_step, current_pos[1], current_pos[2]
    d = current_pos[0], current_pos[1]-x_step, current_pos[2]
    e = current_pos[0], current_pos[1], current_pos[2]-z_step
    f = current_pos[0], current_pos[1], current_pos[2]+z_step

    n_list = np.array([a, b, c, d, e, f])
    # print n_list

    return n_list

def findSuggestion(image_src, dump_path, db, geo_info, seg_dir):

    my_object = SalientObject.SalientObjectDetection(image_src, prediction_stage=False, segment_dump=seg_dir)
    
    comp_map, feature_v, face_v, view_c = my_object.get_photograph_composition()

    new_map = my_object.getNewSaliencyMap()
    faceInfo = my_object.getFaceInfo()

    model_dumps_path = db + "/models/"

    comp_model_path = model_dumps_path + "comp/"
    dump_scalar = comp_model_path + "scalar.pkl"
    dump_pca = comp_model_path + "pca.pkl"
    dump_svm = comp_model_path + "svm.pkl"

    regressor = joblib.load(dump_svm)
    scalar = joblib.load(dump_scalar)
    pca = joblib.load(dump_pca)

    # form feature
    X = []
    X.extend(feature_v)
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

    best_pos = 0, 0, 100
    orig_pos = 0, 0, 100
    best_score = pred[0][0]

    if len(faceInfo) > 0:
        best_pos, best_score, faceInfo, orig_pos = cutils.find_best_position(new_map, faceInfo, view_c, regressor, scalar, pca)       
        print best_pos

    return best_pos, orig_pos, faceInfo, pred, best_score

def plot_suggestion(best_pos, orig_pos, faces, inframe, pred, height, width):
    # img = cv2.imread(infile)
    if len(faces) > 0:
        print orig_pos
        faces_ = faces+[orig_pos[1], orig_pos[0], 0, 0]
        print faces_
        for (x,y,w,h) in faces_:
            cv2.rectangle(inframe,(x,y),(x+w,y+h),(255, 0, 0),2)

        faces_ = (faces*best_pos[2]/100).astype(np.int)
        faces_ = faces_+[best_pos[1], best_pos[0], 0, 0]
        print faces_
        for (x,y,w,h) in faces_:
            cv2.rectangle(inframe,(x,y),(x+w,y+h),(0, 255, 0),2)

    # width of meter
    dx = width/20
    # height of meter
    dy = int(0.6*height)

    xstart = dx
    ystart = dx + dy
    total = dy
    half = total/2
    mid = dy/2 + dx

    # draw a rectangle for visibility
    cv2.rectangle(inframe, (dx-1, dx-1), (dx+dx/2+1, ystart+1), (255, 255, 255), 1)
    cv2.rectangle(inframe, (dx-3, dx-3), (dx+dx/2+3, ystart+3), (0, 0, 0), 1)
    value = total - int(dy*pred[0][0]) + dx

    R, G, B = 255, 0, 0
    while (ystart > mid):
        color = (B, G, R)
        cv2.line(inframe, (xstart, ystart), (xstart+dx/2, ystart), color)

        R = 255*(ystart-mid)/half
        # G = 0
        B = 255*(half - ystart + mid)/half
        ystart -= 1

    R, G, B = 0, 0, 255
    while (ystart > value):
        color = (B, G, R)
        cv2.line(inframe, (xstart, ystart), (xstart+dx/2, ystart), color)

        # R = 0
        G = 255*(mid - ystart)/half
        B = 255*(half + ystart - mid)/half
        ystart -= 1

    return inframe


def process_image_list(w_dir, image_dir, dump_path, model_path, seg_dir, file_list='image.list'):
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    pred_list = w_dir + "/pos_prediction.list"
    fp_pred = open(pred_list, 'w')

    image_list = w_dir + '/' + file_list
    fp_image_list = open(image_list, 'r')

    f_geo = w_dir + "/geo.info"
    f_cam = w_dir + "/camera.settings"
    f_aes = w_dir + "/aesthetic.scores"
    geo_list = np.loadtxt(f_geo)
    a_score = np.loadtxt(f_aes)

    i = 0
    
    height = 480
    width = 640
    for image_name in fp_image_list:
        image_name = image_name.rstrip("\r\n")
        infile = image_dir + image_name
        outfile = dump_path + image_name

        print infile

        timer = time.time()
        frame = cv2.imread(infile)
        height, width, depth = frame.shape
        best_pos, orig_pos, faces, pred, best_score = findSuggestion(infile, dump_path, model_path, geo_list[i], seg_dir)
        
        n_frame = plot_suggestion(best_pos, orig_pos, faces, frame, pred, height, width)
        
        fp_pred.write("{0}\tActual\tPredicted\n".format(image_name))
        fp_pred.write('Score :\t%.2f\t%.2f\t%.2f\n' %(a_score[i], pred[0][0], best_score))

        print "Total run time = ", time.time() - timer

        cv2.imwrite(outfile, n_frame)

        i += 1


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path, seg_cluster_DB"
        sys.exit(0)

    w_dir = sys.argv[1]
    m_path = sys.argv[2] # dumps of clusters
    # w_dir = "/home/vyzuer/windows/Project/Flickr-code/pos_results/param2/"

    seg_dir = m_path + "cluster_dump/"
    db_src = w_dir + "/ImageDB/"
    dump_path = w_dir + "/dump_ImageDB/"

    process_image_list(w_dir, db_src, dump_path, file_list='image.list', model_path=m_path, seg_dir=seg_dir)

