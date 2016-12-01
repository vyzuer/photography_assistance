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

def modify_map(new_map, faces, x_start):
    seg_map2 = np.copy(new_map)
    for (x, y, w, h) in faces:
        # print x, y, w, h
        # x - horizontal position
        # y - vertical position
        # w - width
        # h - height
        for i in range(w):
            for j in range(h):
                seg_map2[y+j][x_start+x+i] = 1.0

    return seg_map2

def findSuggestion(image_src, dump_path, db, geo_info, env_info):

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

    max_pos = 0
    if len(faceInfo) > 0:

        img_height, img_width = new_map.shape
        # find bounding box for faces
        min_x, min_y = 10000, 10000
        max_x, max_y = 0, 0
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

        # print min_x, min_y, max_x, max_y
        box_h = max_y - min_y
        box_w = max_x - min_x
        # print box_h, box_w

        # delete faces first
        for (x, y, w, h) in faceInfo:
                # x - horizontal position
                # y - vertical position
                # w - width
                # h - height
                for i in range(w):
                    for j in range(h):
                        new_map[y+j][x+i] = 0.0

        x_start = -1*min_x
        y_start = min_y
        x_step = int(img_width/50)

        # iterate in grid
        n_step = int(img_width - box_w)/x_step
        # print n_step, x_step, box_w, box_h
        max_pred = 0.0
        max_pos = 0
        for ii in range(n_step):

            # print x_start
            modified_map = modify_map(new_map, faceInfo, x_start)
            # print modified_map.shape
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

            # print pred[0]
            if max_pred < pred[0][0]:
                max_pred = pred[0][0]
                max_pos = x_start

            x_start += x_step

        # print max_pred, max_pos

    # parameter prediction goes here....
    # exposure
    dump_f_scalar = db + "/f_scalar_ev.pkl"
    dump_t_scalar = db + "/t_scalar_ev.pkl"
    dump_pca = db + "/pca_ev.pkl"
    dump_svr = db + "/svr_ev.pkl"

    regressor = joblib.load(dump_svr)
    f_scalar = joblib.load(dump_f_scalar)
    t_scalar = joblib.load(dump_t_scalar)
    pca = joblib.load(dump_pca)

    X = []
    X.extend(env_info)

    X1 = pca.transform(view_c)
    # print X1[0]
    X.extend(X1[0])
    X.extend(geo_info)

    X = np.asarray(X)
    # print X.shape
    X = np.reshape(X, (1, -1))
    # print X.shape

    X = f_scalar.transform(X)
    X = preprocessing.normalize(X, norm='l2')

    ev = regressor.predict(X)
    ev_value = t_scalar.inverse_transform(ev)
    print ev_value
        
    # aperture value
    dump_f_scalar = db + "/f_scalar_ap.pkl"
    dump_t_scalar = db + "/t_scalar_ap.pkl"
    dump_pca = db + "/pca_ap.pkl"
    dump_svr = db + "/svr_ap.pkl"

    regressor = joblib.load(dump_svr)
    f_scalar = joblib.load(dump_f_scalar)
    t_scalar = joblib.load(dump_t_scalar)
    pca = joblib.load(dump_pca)

    X = []
    X.extend(env_info)

    # print face_v
    X.extend(face_v)
    X.extend(ev_value)

    X1 = pca.transform(view_c)
    # print X1[0]
    X.extend(X1[0])
    X.extend(geo_info)

    X = np.asarray(X)
    # print X.shape
    X = np.reshape(X, (1, -1))
    # print X.shape

    X = f_scalar.transform(X)
    X = preprocessing.normalize(X, norm='l2')

    ap = regressor.predict(X)
    ap_value = t_scalar.inverse_transform(ap)
    print ap_value
    
    # shutter speed value
    dump_f_scalar = db + "/f_scalar_ss.pkl"
    dump_t_scalar = db + "/t_scalar_ss.pkl"
    dump_pca = db + "/pca_ss.pkl"
    dump_svr = db + "/svr_ss.pkl"

    regressor = joblib.load(dump_svr)
    f_scalar = joblib.load(dump_f_scalar)
    t_scalar = joblib.load(dump_t_scalar)
    pca = joblib.load(dump_pca)

    X = []
    X.extend(env_info)

    # print face_v
    X.extend(face_v)
    X.extend(ev_value)

    X1 = pca.transform(view_c)
    # print X1[0]
    X.extend(X1[0])
    X.extend(geo_info)
    X.extend(ap_value)

    X = np.asarray(X)
    # print X.shape
    X = np.reshape(X, (1, -1))
    # print X.shape

    X = f_scalar.transform(X)
    X = preprocessing.normalize(X, norm='l2')

    ss = regressor.predict(X)
    print ss
    ss_value = t_scalar.inverse_transform(ss)
    print ss_value

    ss = int(1/math.pow(2, ss_value[0]))
    ap = round(ap_value[0],1)
    ev = ev_value[0]
    print ss
    print ap
    print ev

    iso = int(100*math.pow(2, math.log(ap*ap*ss, 2) - ev))
    print iso

    return max_pos, faceInfo, ss, ap, ev, iso, pred

def plot_suggestion(max_pos, faces, inframe, ss, ap, ev, iso, pred, height, width):
    # img = cv2.imread(infile)
    for (x,y,w,h) in faces:
        cv2.rectangle(inframe,(x,y),(x+w,y+h),(255, 0, 0),2)
        cv2.rectangle(inframe,(max_pos+x,y),(max_pos+x+w,y+h),(0, 255, 0),2)

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

    x = xstart #position of text
    y = height - dx #position of text
    # font1 =cv2.FONT_HERSHEY_SIMPLEX
    font1 =cv2.FONT_HERSHEY_TRIPLEX
    font2 =cv2.FONT_HERSHEY_PLAIN
    font3 =cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(inframe, "Quality - %.2f" % pred[0][0], (x,y), font, 2, (0, 0, 255), 2)
    # y = 70
    # cv2.putText(inframe, "Exposure - %.2f" % ev, (x,y), font, 2, (0, 0, 255), 2)
    # y = 100
    cv2.putText(inframe, "ISO : %d" % iso, (x,y), font1, 0.6, (0, 0, 255), 1)
    y -= dx
    cv2.putText(inframe, "SS : 1/%d" % ss, (x,y), font1, 0.6, (0, 0, 255), 1)
    y -= dx
    cv2.putText(inframe, "AP : %.1f" % ap, (x,y), font1, 0.6, (0, 0, 255), 1)

    return inframe


def testing(w_dir, image_dir, dump_path, model_path, file_list='image.list'):
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    image_list = w_dir + '/' + file_list
    fp_image_list = open(image_list, 'r')

    f_geo = w_dir + "/geo.info"
    f_env = w_dir + "/weather.info"
    geo_list = np.loadtxt(f_geo)
    env_list = np.loadtxt(f_env)

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
        max_pos, faces,  ss, ap, ev, iso, pred = findSuggestion(infile, dump_path, model_path, geo_list[i], env_list[i])
        
        # plot_suggestion(max_pos, faces, infile, outfile)
        n_frame = plot_suggestion(max_pos, faces, frame, ss, ap, ev, iso, pred, height, width)
        
        print "Total run time = ", time.time() - timer

        cv2.imwrite(outfile, n_frame)

        i += 1

# w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/"
# db_src = w_dir + "ImageDB_640/"
# dump_path = w_dir + "dump_Image_640/"

if __name__ == '__main__':

    # if len(sys.argv) != 2:
    #     print "Usage : dataset_path"
    #     sys.exit(0)

    # w_dir = sys.argv[1]
    w_dir = "/home/vyzuer/windows/Project/Flickr-code/pos_results/param2/"

    m_path = "/mnt/windows/DataSet-YsR/merlion/"
    # w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/"
    db_src = w_dir + "/ImageDB/"
    dump_path = w_dir + "/dump_ImageDB/"

    testing(w_dir, db_src, dump_path, file_list='image.list', model_path=m_path)

