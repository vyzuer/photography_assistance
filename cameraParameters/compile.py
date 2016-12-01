import svm
import numpy as np
import math
import os
import sys

train_file = "fv.tr"
test_file = "fv.te"

db_path = './DB/'
camera_settings = 'settings.list'
feature_file = 'fv.list'

if len(sys.argv) < 4:
    sys.exit('Usage: %s dump_path cam.settings fv.list' % sys.argv[0])

db_path = sys.argv[1]
camera_settings = sys.argv[2]
feature_file = sys.argv[3]

RATIO = 0.8


def dumpEV(setting_file):
    target_file = 'ev.score'
    lum_file = 'luminance.score'
    t_file = open(target_file, 'w')
    l_file = open(lum_file, 'w')
    data = np.loadtxt(setting_file, unpack=True)
    dim, n_samples = data.shape

    for i in range(n_samples):
        S = data[0][i]
        A = data[1][i]
        ISO = data[2][i]
        # print A, S, ISO
        e_value = math.log(A*A/S, 2) - math.log(ISO/100, 2)
        # print e_value

        lum_value = math.pow(2, e_value-3)

        t_file.write('{0}\n'.format(e_value))
        l_file.write('{0}\n'.format(lum_value))

    t_file.close()

    return target_file


def dumpTrainTestFile(feature_file, target_file, test_file, train_file):
    train_file = open(train_file, 'w')
    test_file = open(test_file, 'w')

    features = np.loadtxt(feature_file, unpack=True)
    targets = np.loadtxt(target_file, unpack=True)

    dim, n_samples = features.shape
    f_file = train_file
    # print n_samples
    training_samples = int(n_samples*RATIO)
    # print training_samples
    for i in range(n_samples):
        f_file.write('{0}'.format(targets[i]))
        for j in range(dim):
            f_file.write(' {0}:{1}'.format(j+1, features[j][i]))
        f_file.write('\n')

        if i > training_samples :
            f_file = test_file

    train_file.close()
    test_file.close()


def run(train_file, test_file):
    svm.svm(train_file, test_file)

# generateDB(db_dir)    

os.chdir(db_path)

target_file = dumpEV(camera_settings)

# dumpTrainTestFile(feature_file, target_file, test_file, train_file)

# run(train_file, test_file)    

