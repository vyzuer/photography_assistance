#!/bin/csh 


set dataset_path = /home/vyzuer/DUMPS/test_camera_motion/
set model_path = /home/vyzuer/DUMPS/offline.1/

# foreach location (`ls $dataset_path`)
foreach location (liberty)
# foreach location (arcde colognecathedral esplanade floatMarina gatewayofindia eifel indiagate leaningtower liberty merlion taj vatican)
    echo $location
    python cameraMotion.py "${dataset_path}${location}/" "${model_path}${location}/"
end


