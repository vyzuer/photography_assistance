import SalientObject
import sys
import time
import os
import glob
import numpy as np
import shutil


def dump_features(comp_map, fp):

    comp_map = np.reshape(comp_map, -1)
    np.savetxt(fp, np.atleast_2d(comp_map), fmt='%.8f')


def generateDB(image_src, dump_path, fp):

    my_object = SalientObject.SalientObjectDetection(image_src)

    map_dumps = dump_path + "map_dumps/"
    if not os.path.exists(map_dumps):
        os.makedirs(map_dumps)
    my_object.plot_maps(map_dumps)

    comp_map, feature_v, face_v, view_c = my_object.get_photograph_composition()

    dump_features(comp_map, fp)
    

def process_dataset(dataset_path, dump_path):
    # browsing the directory

    image_dir = dataset_path + "ImageDB/"
    
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    # Copy required files to dump path
    image_list = dataset_path + "image.list"
    shutil.copy(image_list, dump_path)

    fv_dump_path = dump_path + "feature_dumps/"
    if not os.path.exists(fv_dump_path):
        os.makedirs(fv_dump_path)

    comp_map_fv = fv_dump_path + "comp_map.list"
    if os.path.isfile(comp_map_fv):
        os.remove(comp_map_fv)

    fp = file(comp_map_fv, 'w')

    fp_image_list = open(image_list, 'r')

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        infile = image_dir + image_name

        # print infile

        timer = time.time()

        generateDB(infile, dump_path, fp)
        
        print "Total run time = ", time.time() - timer

    fp.close()


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    process_dataset(dataset_path, dump_path)


