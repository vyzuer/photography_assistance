#!/bin/csh 


set dataset_path = /home/vyzuer/DUMPS/tests_pos/
set model_path = /home/vyzuer/DUMPS/offline.1/

# foreach location (floatMarina taj)
foreach location (arcde colognecathedral esplanade floatMarina gatewayofindia eifel indiagate leaningtower liberty merlion taj vatican)
    echo $location
    python positionReco.py "${dataset_path}${location}/" "${model_path}${location}/"
end


