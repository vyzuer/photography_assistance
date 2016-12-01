import SalientObject
import sys
import time
import os
import glob
import numpy as np


def generateDB(image_src, dump_path, gen_all=False):

    my_object = SalientObject.SalientObjectDetection(image_src)
    # my_object.plot_maps(dump_path)
    my_object.process_segments(dump_path)
    
    comp_map, feature_v, face_v, view_c = my_object.get_photograph_composition()

    return comp_map, feature_v, face_v, view_c

def testing(w_dir, image_dir, dump_path, file_list='image.list'):
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    # browsing the directory
    feature_file = dump_path + "feature.fv"
    if os.path.isfile(feature_file):
        os.remove(feature_file)

    face_fv = dump_path + "face.fv"
    if os.path.isfile(face_fv):
        os.remove(face_fv)

    view_fv = dump_path + "view.fv"
    if os.path.isfile(view_fv):
       os.remove(view_fv)

    comp_map_fv = dump_path + "comp_map.fv"
    if os.path.isfile(comp_map_fv):
        os.remove(comp_map_fv)

    # f_handle1 = file(feature_file, 'a')
    # f_handle2 = file(face_fv, 'a')
    # f_handle3 = file(view_fv, 'a')
    # f_handle4 = file(comp_map_fv, 'a')

    f_handle1 = file(feature_file, 'w')
    f_handle2 = file(face_fv, 'w')
    f_handle3 = file(view_fv, 'w')
    f_handle4 = file(comp_map_fv, 'w')

    image_list = w_dir + '/' + file_list
    fp_image_list = open(image_list, 'r')

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\r\n")
        infile = image_dir + image_name

        print infile

        timer = time.time()
        comp_map, feature_v, face_v, view_c = generateDB(infile, dump_path, gen_all=True)
        
        # dump feature vector
        np.savetxt(f_handle1, np.atleast_2d(feature_v), fmt='%.8f')
        np.savetxt(f_handle2, np.atleast_2d(face_v), fmt='%.8f')
        np.savetxt(f_handle3, np.atleast_2d(view_c), fmt='%.8f')

        comp_map = np.reshape(comp_map, -1)
        np.savetxt(f_handle4, np.atleast_2d(comp_map), fmt='%.8f')

        print "Total run time = ", time.time() - timer

    f_handle1.close()
    f_handle2.close()
    f_handle3.close()
    f_handle4.close()


# db_src = "../testing/images_1/"
# dump_path = "../testing/dump_1/"
# w_dir = "../testing/images_1/"

# w_dir = "/home/vyzuer/Project/Flickr-YsR/esplanade/"
# # w_dir = "/home/vyzuer/Project/Flickr-YsR/floatMarina/"
# db_src = w_dir + "ImageDB_640/"
# dump_path = w_dir + "dump_ImageDB_640/"

# w_dir = "/home/yogesh/Project/EigenRules/"
# db_src = w_dir + "ImageDB/"
# dump_path = w_dir + "dump_ImageDB/"

# w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/"
# db_src = w_dir + "ImageDB_640/"
# dump_path = w_dir + "dump_Image_640/"

# w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/Good_Image_DB/"
# db_src = w_dir + "ImageDB_640/"
# dump_path = w_dir + "dump_640/"

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "Usage : dataset_path"
        sys.exit(0)

    w_dir = sys.argv[1]
    # w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/"
    db_src = w_dir + "/ImageDB/"
    dump_path = w_dir + "/dump_ImageDB/"

    testing(w_dir, db_src, dump_path)

