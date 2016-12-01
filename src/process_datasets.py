import SalientObject
import sys
import time
import os
import glob
import numpy as np
import shutil


def dump_features(comp_map, feature_v, face_v, view_c, f1, f2, f3, f4):

    np.savetxt(f1, np.atleast_2d(feature_v), fmt='%.8f')
    np.savetxt(f2, np.atleast_2d(face_v), fmt='%.8f')
    np.savetxt(f3, np.atleast_2d(view_c), fmt='%.8f')

    comp_map = np.reshape(comp_map, -1)
    np.savetxt(f4, np.atleast_2d(comp_map), fmt='%.8f')


def generateDB(image_src, dump_path, f1, f2, f3, f4):

    my_object = SalientObject.SalientObjectDetection(image_src)

    # seg_dumps = dump_path + "segment_dumps/"
    # if not os.path.exists(seg_dumps):
    #     os.makedirs(seg_dumps)
    # my_object.process_segments(seg_dumps)

    map_dumps = dump_path + "map_dumps/"
    if not os.path.exists(map_dumps):
        os.makedirs(map_dumps)
    my_object.plot_maps(map_dumps)

    comp_map, feature_v, face_v, view_c = my_object.get_photograph_composition()

    dump_features(comp_map, feature_v, face_v, view_c, f1, f2, f3, f4)
    

def process_dataset(dataset_path, dump_path):
    # browsing the directory

    image_dir = dataset_path + "ImageDB/"
    
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    # Copy required files to dump path
    image_list = dataset_path + "image.list"
    shutil.copy(image_list, dump_path)
    img_details = dataset_path + "images.details"
    shutil.copy(img_details, dump_path)
    img_aesthetic_score = dataset_path + "aesthetic.scores"
    shutil.copy(img_aesthetic_score, dump_path)
    weather_data = dataset_path + "weather.info"
    shutil.copy(weather_data, dump_path)
    geo_info = dataset_path + "geo.info"
    shutil.copy(geo_info, dump_path)
    cam_info = dataset_path + "camera.settings"
    shutil.copy(cam_info, dump_path)

    fv_dump_path = dump_path + "feature_dumps/"
    if not os.path.exists(fv_dump_path):
        os.makedirs(fv_dump_path)

    feature_file = fv_dump_path + "feature.list"
    if os.path.isfile(feature_file):
        os.remove(feature_file)

    face_fv = fv_dump_path + "face.list"
    if os.path.isfile(face_fv):
        os.remove(face_fv)

    view_fv = fv_dump_path + "view.list"
    if os.path.isfile(view_fv):
       os.remove(view_fv)

    comp_map_fv = fv_dump_path + "comp_map.list"
    if os.path.isfile(comp_map_fv):
        os.remove(comp_map_fv)

    f1 = file(feature_file, 'w')
    f2 = file(face_fv, 'w')
    f3 = file(view_fv, 'w')
    f4 = file(comp_map_fv, 'w')

    fp_image_list = open(image_list, 'r')

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        infile = image_dir + image_name

        # print infile

        timer = time.time()

        generateDB(infile, dump_path, f1, f2, f3, f4)
        
        print "Total run time = ", time.time() - timer

    f1.close()
    f2.close()
    f3.close()
    f4.close()


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    process_dataset(dataset_path, dump_path)


