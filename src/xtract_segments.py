import SalientObject
import sys
import time
import os
import glob
import numpy as np
import shutil


def generateDB(image_src, dump_path):

    my_object = SalientObject.SalientObjectDetection(image_src)

    seg_dumps = dump_path + "segment_dumps/"
    if not os.path.exists(seg_dumps):
        os.makedirs(seg_dumps)
    my_object.process_segments(seg_dumps)

    map_dumps = dump_path + "map_dumps/"
    if not os.path.exists(map_dumps):
        os.makedirs(map_dumps)
    my_object.plot_maps(map_dumps)

    

def testing(w_dir, image_dir, dump_path, file_list='image.list'):
    # browsing the directory
    
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    image_list = w_dir + file_list

    # Copy required files to dump path
    shutil.copy(image_list, dump_path)
    img_details = w_dir + "images.details"
    shutil.copy(img_details, dump_path)
    img_aesthetic_score = w_dir + "aesthetic.scores"
    shutil.copy(img_aesthetic_score, dump_path)
    weather_data = w_dir + "weather.info"
    shutil.copy(weather_data, dump_path)

    fp_image_list = open(image_list, 'r')

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        infile = image_dir + image_name

        print infile

        timer = time.time()
        generateDB(infile, dump_path)
        
        print "Total run time = ", time.time() - timer


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    w_dir = sys.argv[1]
    dump_path = sys.argv[2]

    db_src = w_dir + "ImageDB/"

    testing(w_dir, db_src, dump_path)


# w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/"
# db_src = w_dir + "ImageDB_640/"
# dump_path = w_dir + "segments_dump_640/"

# w_dir = "/home/yogesh/Project/Flickr-YsR/merlionImages/Good_Image_DB/"
# db_src = w_dir + "ImageDB_640/"
# dump_path = w_dir + "segment_dump/"


