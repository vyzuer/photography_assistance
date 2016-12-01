#!/bin/csh 


set dataset_path = /home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/
set model_path = /home/vyzuer/DUMPS/offline.1/merlion/
python positionReco.py "${dataset_path}" "${model_path}"


