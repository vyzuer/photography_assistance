import svm
import numpy as np
import math
import os
import sys

# db_path = './DB/'
# camera_settings = 'camera.settings'
# feature_file = 'fv_aperture.list'
# weather_info = 'weather.info'
# ev_score = 'ev.score'

if len(sys.argv) < 7:
    sys.exit('Usage: %s db_path camera_settings weather ev_score' % sys.argv[0])

db_path = sys.argv[1]
camera_settings = sys.argv[2]
ev_score = sys.argv[3]
weather_info = sys.argv[4]
fv_file = sys.argv[5]
target_file = sys.argv[6]

def gen_features(camera_settings, ev_score, weather_info, fv_file):

    # load features
    cam_data = np.loadtxt(camera_settings)
    ev_data = np.loadtxt(ev_score)
    weather_data = np.loadtxt(weather_info)
    # print cam_data.shape
    # print weather_data.shape
    ev_data = np.reshape(ev_data, (-1, 1))
    # print ev_data.shape

    time_data = weather_data[:, 0:3]
    # print time_data.shape
    focal_length = cam_data[:, 3]
    focal_length = np.reshape(focal_length, (-1, 1))
    # print focal_length.shape
    
    X = np.hstack([time_data, focal_length, ev_data])

    np.savetxt(fv_file, X, fmt='%0.6f')


def gen_targets(camera_settings, target_file):
    
    # load features
    cam_data = np.loadtxt(camera_settings)
    aperture = cam_data[:, 1]

    np.savetxt(target_file, aperture, fmt='%0.6f')


os.chdir(db_path)

gen_features(camera_settings, ev_score, weather_info, fv_file)

gen_targets(camera_settings, target_file)


