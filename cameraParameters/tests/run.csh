#!/bin/csh -f

set db_path = "./DB/"
set data = "./DB/weather.info"
set target = "./DB/camera.settings"
set a_score = "./DB/aesthetic.scores"

echo "filtering bad features..."
# ./filter_features.csh $db_path $data $target $a_score
echo "done."

echo "evaluate ev score..."
python compile.py $db_path "cam.settings" "fv.list"
echo "done."


echo "learning model for camera parameters..."

python classify.py "./DB/fv.list" "./DB/ev.score"

echo "done."
