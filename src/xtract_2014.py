import SalientObject
import sys
import time
import os
import glob
import numpy as np


def generateDB(image_src, dump_path, gen_all=False):

    my_object = SalientObject.SalientObjectDetection(image_src)
    my_object.plot_maps(dump_path)
    
    composition_vector, feature_vec = my_object.get_photograph_composition()

    if gen_all == True:
        cv_1, fv_1 = my_object.get_photograph_composition_full()

        cv_2, fv_2 = my_object.get_photograph_composition_avg()

        cv_3, fv_3 = my_object.get_photograph_composition_full_avg()

        my_object.plot_composition_map(dump_path)

    if gen_all == True:
        return composition_vector, feature_vec, cv_1, fv_1, cv_2, fv_2, cv_3, fv_3
    else:
        return composition_vector, feature_vec

def testing(w_dir, image_dir, dump_path, file_list='image.list', gen_all=False, clean=True):
    # browsing the directory
    feature_file = w_dir + "composition.fv"
    comp_file = w_dir + "composition_map.fv"
    if os.path.isfile(feature_file) and clean == True:
        os.remove(feature_file)

    if os.path.isfile(comp_file) and clean == True:
        os.remove(comp_file)

    f_handle = file(feature_file, 'a')
    c_handle = file(comp_file, 'a')

    f_handle_1 = None
    f_handle_2 = None
    f_handle_3 = None

    c_handle_1 = None
    c_handle_2 = None
    c_handle_3 = None

    if gen_all == True:
        f_1 = w_dir + "composition_full.fv"
        if os.path.isfile(f_1) and clean == True:
            os.remove(f_1)
            
        f_handle_1 = file(f_1, 'a')

        f_2 = w_dir + "composition_avg.fv"
        if os.path.isfile(f_2) and clean == True:
            os.remove(f_2)
            
        f_handle_2 = file(f_2, 'a')

        f_3 = w_dir + "composition_full_avg.fv"
        if os.path.isfile(f_3) and clean == True:
            os.remove(f_3)
            
        f_handle_3 = file(f_3, 'a')

        c_1 = w_dir + "composition_full_map.fv"
        if os.path.isfile(c_1) and clean == True:
            os.remove(c_1)
            
        c_handle_1 = file(c_1, 'a')

        c_2 = w_dir + "composition_avg_map.fv"
        if os.path.isfile(c_2) and clean == True:
            os.remove(c_2)
            
        c_handle_2 = file(c_2, 'a')

        c_3 = w_dir + "composition_full_avg_map.fv"
        if os.path.isfile(c_3) and clean == True:
            os.remove(c_3)
            
        c_handle_3 = file(c_3, 'a')

    image_list = w_dir + file_list
    fp_image_list = open(image_list, 'r')

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        infile = image_dir + image_name

        print infile

        timer = time.time()
        comp_vec = None
        if gen_all == True:
            comp_vec, fv, cv_1, fv_1, cv_2, fv_2, cv_3, fv_3 = generateDB(infile, dump_path, gen_all=True)
        else:
            comp_vec, fv = generateDB(infile, dump_path)
        
        # dump feature vector

        np.savetxt(f_handle, np.atleast_2d(fv), fmt='%.8f')
        np.savetxt(c_handle, np.atleast_2d(np.ravel(comp_vec)), fmt='%.8f')

        if gen_all == True:
            np.savetxt(f_handle_1, np.atleast_2d(fv_1), fmt='%.8f')
            np.savetxt(c_handle_1, np.atleast_2d(np.ravel(cv_1)), fmt='%.8f')

            np.savetxt(f_handle_2, np.atleast_2d(fv_2), fmt='%.8f')
            np.savetxt(c_handle_2, np.atleast_2d(np.ravel(cv_2)), fmt='%.8f')

            np.savetxt(f_handle_3, np.atleast_2d(fv_3), fmt='%.8f')
            np.savetxt(c_handle_3, np.atleast_2d(np.ravel(cv_3)), fmt='%.8f')

        print "Total run time = ", time.time() - timer

    f_handle.close()
    c_handle.close()
    if gen_all == True:
        f_handle_1.close()
        f_handle_2.close()
        f_handle_3.close()

        c_handle_1.close()
        c_handle_2.close()
        c_handle_3.close()

# w_dir = "../testing/DB1/"

# w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/"
# w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/Good_Image_DB/"
w_dir = "/home/yogesh/Project/Flickr-YsR/InterestingDB/Raw_2014_Info/"
# w_dir = "/home/yogesh/Project/Flickr-YsR/InterestingDB/Raw_Info/"

# w_dir = "/home/yogesh/Project/EigenRules/"    
 
db_src = w_dir + "ImageDB/"
dump_path = w_dir + "dump_ImageDB/"

testing(w_dir, db_src, dump_path, gen_all = True, clean = False)

