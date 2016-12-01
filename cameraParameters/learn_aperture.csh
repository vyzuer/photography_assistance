#!/bin/csh -f

set scripts_path = "/home/vyzuer/Copy/Flickr-code/PhotographyAssistance/preProcess/"
set dump_path = "/home/vyzuer/Copy/Flickr-code/PhotographyAssistance/cameraParameters/apertureDB/"

set db_path = "/home/vyzuer/Project/Flickr-YsR/merlionImages/Good_Image_DB/"
set weather_db = "/home/vyzuer/Copy/Flickr-code/weatherDB/"
set image_db = "${db_path}/ImageDB/"
set images_details = "${db_path}/images.details"


set weather_info = 'weather.info'
set fv_file = 'fv_aperture.list'
set cam_settings = "camera.settings"
set ev_score = "ev.score"
set a_target = "ap.target"

# get weather info from weather DB
# ${scripts_path}genWeatherData.csh ${db_path} ${weather_db} ${dump_path}

# get camera settings for aperture learning
# ${scripts_path}dumpCameraParameters.csh ${db_path} ${dump_path}

# dump ev score for the images
echo "Generating EV score..."
python dump_ev.py $dump_path $cam_settings $ev_score

# generate feature file for aperture learning
echo "Generating aperture features..."
python generate_aperture_features.py $dump_path $cam_settings $ev_score $weather_info $fv_file $a_target

echo "Learning Aperture Model..."
python aperture_classify.py $dump_path $fv_file $a_target

