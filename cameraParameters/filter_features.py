import numpy as np
import math
import os
import sys


def dumpEV(f_ev, f_cam_1):
    t_file = open(f_ev, 'w')
    data = np.loadtxt(f_cam_1)
    n_samples, dim = data.shape

    for i in range(n_samples):
        S = data[i][0]
        A = data[i][1]
        ISO = data[i][2]
        # print A, S, ISO
        e_value = math.log(A*A/S, 2) - math.log(ISO/100, 2)
        # print e_value

        t_file.write('{0}\n'.format(e_value))

    t_file.close()


def filter_f(f_file, map_file, outfile):
    print outfile
    fp = open(outfile, 'w')
    X = np.loadtxt(f_file, delimiter=' ', dtype=np.str)
    y = np.loadtxt(map_file)

    n_samples = len(X)
    for i in range(n_samples):
        if y[i] == 1:
            np.savetxt(fp, np.atleast_2d(X[i]), fmt='%s')
            # fp.write(X[i])

    fp.close()


def filter_features(db_path, dump_dir):
    # db = "/home/yogesh/Project/Flickr-YsR/"
    dump_path = db_path + dump_dir

    f_comp = dump_path + "/comp_map.list"
    f_view = dump_path + "/view.list"
    f_feature = dump_path + "/feature.list"
    f_face = dump_path + "/face.list"

    f_comp_1 = dump_path + "/comp_map_1.list"
    f_view_1 = dump_path + "/view_1.list"
    f_feature_1 = dump_path + "/feature_1.list"
    f_face_1 = dump_path + "/face_1.list"

    f_geo = db_path + "/geo.info"
    f_cam = db_path + "/camera.settings"
    f_env = db_path + "/weather.info"
    f_score = db_path + "/aesthetic.scores"

    f_geo_1 = db_path + "/geo_1.info"
    f_cam_1 = db_path + "/camera_1.settings"
    f_env_1 = db_path + "/weather_1.info"
    f_score_1 = db_path + "/aesthetic_1.scores"


    map_file = db_path + "/map.list"

    filter_f(f_comp, map_file, f_comp_1)
    filter_f(f_view, map_file, f_view_1)
    filter_f(f_feature, map_file, f_feature_1)
    filter_f(f_face, map_file, f_face_1)

    filter_f(f_geo, map_file, f_geo_1)
    filter_f(f_cam, map_file, f_cam_1)
    filter_f(f_env, map_file, f_env_1)
    filter_f(f_score, map_file, f_score_1)

    f_ev = db_path + "/ev.score"
    dumpEV(f_ev, f_cam_1)

if __name__ == "__main__":

    #db_path = "/home/yogesh/Project/Flickr-YsR/floatMarina/"
    # db_path = "/home/yogesh/Project/Flickr-YsR/esplanade/"
    #feature_file = db_path + "dump_ImageDB_640/comp_map.fv"
    # feature_file = db_path + "dump_ImageDB_640/feature.fv"
    # feature_file = "/home/yogesh/Project/Flickr-YsR/merlionImages/dump_ImageDB_640/composition_full_avg.fv"
    #a_score_file = db_path + "aesthetic.scores"

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump"
        sys.exit(0)

    db_path = sys.argv[1]
    dump_dir = sys.argv[2]

    filter_features(db_path, dump_dir)
