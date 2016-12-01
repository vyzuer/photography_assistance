import svm

#db_dir = "/home/scps/myDrive/Copy/Flickr-code/DB/merlionImages/"
#db_dir = "/home/scps/myDrive/Copy/Flickr-code/DB/merlionImages/clusters/0/"
#train_file = db_dir + "features.tr"
#test_file = db_dir + "features.te"

train_file = "GOLD_3April/features.tr.corrected"
test_file = "GOLD_3April/features.te.corrected"

#train_file = "f1.tr"
#test_file = "f1.te"

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

run(train_file, test_file)    

