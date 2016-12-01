import svm

#db_dir = "/home/scps/myDrive/Copy/Flickr-code/DB/merlionImages/"
#db_dir = "/home/scps/myDrive/Copy/Flickr-code/DB/merlionImages/clusters/0/"
#train_file = db_dir + "features.tr"
#test_file = db_dir + "features.te"

def run(train_file, test_file):
    svm.svm(train_file, test_file)

def generateDB(db_path):
    generateDataSet.generateDataSet(db_path)


def processImagsDB(image_dir):
    # browsing the directory
    for infile in glob.glob(os.path.join(image_dir, '*.jpg')):
        #print infile
        test.processImage(infile)


#generateDB(db_dir)    

train_file = "features.tr.corrected"
test_file = "features.te.corrected"

run(train_file, test_file)    

train_file = "fv.tr"
test_file = "fv.te"

run(train_file, test_file)    

