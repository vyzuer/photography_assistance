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

def findSuggestion(image_src, dump_path, db, geo_info, env_info, seg_dir):

    my_object = SalientObject.SalientObjectDetection(image_src, prediction_stage=True, segment_dump=seg_dir)
    t_form, tilt, pan, zoom, rotate = my_object.estimate_affine_parameters()
    
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

        x_step = int(img_width/50)
        y_step = int(img_height/50)
        # z_step = int((x_step+y_step)/2)
        z_step = 5


        x_init = (min_x+max_x)/2
        y_init = (min_y+max_y)/2

        pos_iter = init_pos*[img_height, img_width]
        pos_iter = np.insert(pos_iter, 0, [y_init, x_init], axis=0).astype(np.int)
        pos_iter = np.insert(pos_iter, 2, 100, axis=1)

        # plt.scatter(pos_iter[:, 1], pos_iter[:, 0])
        # plt.xlim(0, img_width)
        # plt.ylim(0, img_height)
        # plt.show()

        # -----> x direction
        # |
        # |
        # V
        # y direction
        # print pos_iter


        # move the origin to center of boundig box
        # print faceInfo
        orig_pos = [y_init, x_init, 0]
        faceInfo = faceInfo-[x_init, y_init, 0, 0]
        # print faceInfo
        best_score = 0.0
        best_pos = y_init, x_init, 100
        cnt = 0
        for pos in (pos_iter):
            current_pos = pos
            change = True

            modified_map, valid = modify_map(new_map, faceInfo, current_pos)
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
            score = regressor.predict_proba(X)

            while change == True:
                change = False

                # check boundary conditions
                if current_pos[0] + box_h > img_height:
                    break

                if current_pos[0] - box_h < 0:
                    break

                if current_pos[1] + box_w > img_width:
                    break

                if current_pos[1] - box_w < 0:
                    break

                if current_pos[2] > 120:
                    break

                # traverse neighbors of current point
                neighbors_list = find_neighbors(current_pos, y_step, x_step, z_step)
                for p in (neighbors_list):
                    cnt += 1
                    modified_map, valid = modify_map(new_map, faceInfo, p)
                    if valid == False:
                        change = False
                        break

                    timer = time.time()
                    fv = find_feature_vector(modified_map)
                    print "run time = ", time.time() - timer

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
                    score_local = regressor.predict_proba(X)

                    if score_local[0][0] > score[0][0] :
                        score = score_local
                        current_pos = p
                        change = True

            if best_score < score[0][0]:
                best_score = score[0][0]
                best_pos = current_pos

        print best_score
        print cnt

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

    return best_pos, orig_pos, faceInfo, ss, ap, ev, iso, pred, tilt, pan, zoom, rotate

def plot_suggestion(best_pos, orig_pos, faces, inframe, ss, ap, ev, iso, pred, height, width):
    # img = cv2.imread(infile)
    if len(faces) > 0:
        faces_ = faces+[orig_pos[1], orig_pos[0], 0, 0]
        for (x,y,w,h) in faces_:
            cv2.rectangle(inframe,(x,y),(x+w,y+h),(255, 0, 0),2)

        faces_ = (faces*best_pos[2]/100).astype(np.int)
        faces_ = faces_+[best_pos[1], best_pos[0], 0, 0]
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


def process_image_list(w_dir, image_dir, dump_path, model_path, seg_dir, file_list='image.list'):
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    pred_list = w_dir + "/cam_prediction.list"
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
        outfile = dump_path + image_name

        print infile

        timer = time.time()
        frame = cv2.imread(infile)
        height, width, depth = frame.shape
        best_pos, orig_pos, faces,  ss, ap, ev, iso, pred, tilt, pan, zoom, rotate = findSuggestion(infile, dump_path, model_path, geo_list[i], env_list[i], seg_dir)
        
        n_frame = plot_suggestion(best_pos, orig_pos, faces, frame, ss, ap, ev, iso, pred, height, width)
        
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
        fp_pred.write('TILT : %.4f\tPAN : %.4f\tZOOM : %.4f\tROTATE : %.4f\n' %(tilt, pan, zoom, rotate))

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

