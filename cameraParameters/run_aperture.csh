#!/bin/csh -f

set db_path = "./apertureDB/"
set data = "./apertureDB/weather.info"
set target = "./apertureDB/camera.settings"
set a_score = "./apertureDB/aesthetic.scores"

echo "filtering bad features..."
./filter_features.csh $db_path $data $target $a_score
echo "done."

echo "evaluate ev score..."
python compile.py $db_path "cam.settings" "fv.list"
echo "done."

echo "learning model for camera parameters..."

# python classify.py "./DB/fv.list" "./DB/luminance.score"
# python classify.py "./DB/fv.list" "./DB/ev.score"

echo "done."
