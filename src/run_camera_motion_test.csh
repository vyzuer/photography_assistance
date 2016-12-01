#!/bin/csh 


set db_path = /home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/
set dump_path = /home/vyzuer/DUMPS/offline.1/merlion/

# set db_path = /mnt/windows/Project/Flickr-YsR/merlionImages/
# set dump_path = /mnt/windows/Project/DUMPS/Segments/merlionImages/

python cameraMotion.py $db_path $dump_path

