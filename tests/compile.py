import generateDataSet
import SalientObject
import sys
import test
import os
import glob
import time

db_dir = "/home/scps/myDrive/Copy/Flickr-code/DB/merlionImages/clusters/0/"
#db_dir = "/home/scps/myDrive/Copy/Flickr-code/DB/merlionImages/"

def generateDB(argv):
    db_path = str(argv[0])
    image_src = str(argv[1])
    a_score = float(argv[2])
    dump_file = str(argv[3])

    fp_feature_file = open(dump_file, 'a')

    my_object = SalientObject.SalientObjectDetection(image_src, a_score)
    #my_object.plotMaps()
    my_object.dumpFeatureVector(fp_feature_file)
    my_object.processSegments(db_path)
    
    fp_feature_file.close()

def testing(image_dir, dump_path):
    # browsing the directory
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    for infile in glob.glob(os.path.join(image_dir, '*.jpg')):
        #print infile
        timer = time.time()
        test.processImage(infile, dump_path)
        print "Total run time = ", time.time() - timer

db_src = "../temp_db/"
dump_path = "../images_temp_res/"
testing(db_src, dump_path)
#generateDB(sys.argv[1:])

