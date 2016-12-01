import svm
import numpy as np
import math
import os
import sys
 
# db_path = './DB/'
# camera_settings = 'settings.list'
# feature_file = 'fv.list'

if len(sys.argv) < 4:
    sys.exit('Usage: %s db_path camera_settings.file ev.file' % sys.argv[0])

db_path = sys.argv[1]
camera_settings = sys.argv[2]
ev_file = sys.argv[3]

def dumpEV(setting_file, target_file):
    t_file = open(target_file, 'w')
    data = np.loadtxt(setting_file, unpack=True)
    dim, n_samples = data.shape

    for i in range(n_samples):
        S = data[0][i]
        A = data[1][i]
        ISO = data[2][i]
        # print A, S, ISO
        e_value = math.log(A*A/S, 2) - math.log(ISO/100, 2)
        # print e_value

        t_file.write('{0}\n'.format(e_value))

    t_file.close()


os.chdir(db_path)

dumpEV(camera_settings, ev_file)


