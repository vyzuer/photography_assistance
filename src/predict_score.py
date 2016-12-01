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


def findSuggestion(image_src, dump_path, db, geo_info, env_info, seg_dir):

    my_object = SalientObject.SalientObjectDetection(image_src)
    # my_object.plot_maps_2(dump_path)
    # my_object.process_segments(dump_path)
    # my_object.estimate_affine_parameters()
    
    comp_map, feature_v, face_v, view_c = my_object.get_photograph_composition()
    print comp_map.shape

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

    # parameter prediction goes here....
    # exposure
    ev_model_path = model_dumps_path + "/ev/"
    dump_f_scalar = ev_model_path + "/f_scalar_ev.pkl"
    dump_t_scalar = ev_model_path + "/t_scalar_ev.pkl"
    dump_pca = ev_model_path + "/pca_ev.pkl"
    dump_svr = ev_model_path + "/svr_ev.pkl"

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
    # print ev_value
        
    # aperture value
    ap_model_path = model_dumps_path + "/ap/"
    dump_f_scalar = ap_model_path + "/f_scalar_ap.pkl"
    dump_t_scalar = ap_model_path + "/t_scalar_ap.pkl"
    dump_pca = ap_model_path + "/pca_ap.pkl"
    dump_svr = ap_model_path + "/svr_ap.pkl"

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
    ap_value = math.pow(2, ap_value[0])
    # print ap_value
    
    # shutter speed value
    ss_model_path = model_dumps_path + "/ss/"
    dump_f_scalar = ss_model_path + "/f_scalar_ss.pkl"
    dump_t_scalar = ss_model_path + "/t_scalar_ss.pkl"
    dump_pca = ss_model_path + "/pca_ss.pkl"
    dump_svr = ss_model_path + "/svr_ss.pkl"

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
    X.extend([ap_value])

    X = np.asarray(X)
    # print X.shape
    X = np.reshape(X, (1, -1))
    # print X.shape

    X = f_scalar.transform(X)
    X = preprocessing.normalize(X, norm='l2')

    ss = regressor.predict(X)
    # print ss
    ss_value = t_scalar.inverse_transform(ss)
    # print ss_value

    ss = int(1/math.pow(2, ss_value[0]))
    ap = round(ap_value,1)
    ev = ev_value[0]
    print ss
    print ap
    print ev

    iso = int(100*math.pow(2, math.log(ap*ap*ss, 2) - ev))
    print iso

    return ss, ap, ev, iso, pred


def process_image_list(w_dir, image_dir, dump_path, model_path, seg_dir, file_list='image.list'):
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    pred_list = w_dir + "/prediction.list"
    fp_pred = open(pred_list, 'w')

    image_list = w_dir + '/' + file_list
    fp_image_list = open(image_list, 'r')

    f_geo = w_dir + "/geo.info"
    f_env = w_dir + "/weather.info"
    f_cam = w_dir + "/camera.settings"
    f_aes = w_dir + "/aesthetic.scores"
    geo_list = np.loadtxt(f_geo)
    env_list = np.loadtxt(f_env)
    cam_settings = np.loadtxt(f_cam)
    a_score = np.loadtxt(f_aes)

    i = 0
    
    height = 480
    width = 640
    for image_name in fp_image_list:
        image_name = image_name.rstrip("\r\n")
        infile = image_dir + image_name

        print infile

        timer = time.time()
        frame = cv2.imread(infile)
        height, width, depth = frame.shape
        ss, ap, ev, iso, pred = findSuggestion(infile, dump_path, model_path, geo_list[i], env_list[i], seg_dir)
        
        ss_0 = cam_settings[i][0]
        ap_0 = cam_settings[i][1]
        iso_0 = cam_settings[i][2]
        ev_0 = math.log(ap_0*ap_0/ss_0, 2) - math.log(iso_0/100, 2)
        fp_pred.write("{0}\tActual\tPredicted\n".format(image_name))
        fp_pred.write('Score :\t%.2f\t%.2f\n' %(a_score[i], pred[0][0]))
        fp_pred.write('EV :\t%.2f\t%.2f\n' %(ev_0, ev))
        fp_pred.write('AP :\t%.2f\t%.2f\n' %(ap_0, ap))
        fp_pred.write('SS :\t1/%d\t1/%d\n' %(1/ss_0, ss))
        fp_pred.write('ISO :\t%d\t%d\n' %(iso_0, iso))
        
        print "Total run time = ", time.time() - timer

        i += 1

    fp_pred.close()
    fp_image_list.close()


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path, seg_cluster_DB"
        sys.exit(0)

    w_dir = sys.argv[1]
    m_path = sys.argv[2] 

    db_src = w_dir + "/ImageDB/"
    dump_path = w_dir + "/dump_ImageDB/"

    process_image_list(w_dir, db_src, dump_path, file_list='image.list', model_path=m_path, seg_dir=w_dir)

