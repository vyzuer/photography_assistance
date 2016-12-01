import SalientObject
import sys
import time
import os
import glob
import numpy as np

db_dir = "/home/scps/myDrive/Copy/Flickr-code/DB/merlionImages/clusters/0/"
#db_dir = "/home/scps/myDrive/Copy/Flickr-code/DB/merlionImages/"

def generateDB(image_src, dump_path):

    my_object = SalientObject.SalientObjectDetection(image_src)
    # my_object.plot_maps(dump_path)
    my_object.process_segments(dump_path)
    
    composition_vector = my_object.get_photograph_composition()

    return composition_vector

def testing(w_dir, image_dir, dump_path, file_list='image.list'):
    # browsing the directory
    feature_file = w_dir + "composition.fv"
    if os.path.isfile(feature_file):
        os.remove(feature_file)

    f_handle = file(feature_file, 'w')

    image_list = w_dir + file_list
    fp_image_list = open(image_list, 'r')

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        infile = image_dir + image_name

        print infile

        timer = time.time()
        comp_vec = generateDB(infile, dump_path)
        
        # dump feature vector
        np.savetxt(f_handle, np.atleast_2d(comp_vec), fmt='%.8f')

        print "Total run time = ", time.time() - timer

    f_handle.close()


db_src = "../testing/images_1/"
dump_path = "../testing/dump_1/"
w_dir = "../testing/"

# w_dir = "/home/yogesh/Project/Flickr-YsR/small_merlion/"
# db_src = w_dir + "ImageDB/"
# dump_path = w_dir + "dump/"

# w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/Good_Image_DB/"
# db_src = w_dir + "ImageDB_640/"
# dump_path = w_dir + "segment_dump/"

testing(w_dir, db_src, dump_path)

