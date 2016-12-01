#!/bin/csh 

set dataset_path = /home/vyzuer/DUMPS/tests/
set model_path = /home/vyzuer/DUMPS/offline.1/

# foreach location (floatMarina taj)
foreach location (arcde colognecathedral esplanade floatMarina gatewayofindia eifel indiagate leaningtower liberty merlion taj vatican)
    echo $location
    python predict_score.py "${dataset_path}${location}/" "${model_path}${location}/"
end

