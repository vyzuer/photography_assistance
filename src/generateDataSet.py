import numpy
import SalientObject
import gc

def generateDataSet(db_path, file_list='image.list', a_score='aesthetic.scores', dump_file='features'):
    image_list = db_path + file_list

    train_data = db_path + dump_file + ".tr"
    test_data = db_path + dump_file + ".te"

    fp_train_file = open(train_data, 'w')
    fp_test_file = open(test_data, 'w')

    fp_feature_file = fp_train_file

    fp_image_list = open(image_list, 'r')
    a_score = db_path + a_score

    aesthetic_scores = numpy.loadtxt(a_score)

    num_samples = len(aesthetic_scores)
    train_samples = int(num_samples*0.8)
    test_samples = num_samples - train_samples

    i = 0

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        image_src = db_path + "/ImageDB/" + image_name
        print image_src

        my_object = SalientObject.SalientObjectDetection(image_src, aesthetic_scores[i])
        my_object.plotMaps()
        my_object.dumpFeatureVector(fp_feature_file)

        i += 1

        if i > train_samples:
            fp_feature_file = fp_test_file
    
    fp_train_file.close()
    fp_test_file.close()
    fp_image_list.close()


