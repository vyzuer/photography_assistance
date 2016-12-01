#!/bin/csh 


set db_path = /home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/
set dump_path = /home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/dump/

# set db_path = /mnt/windows/Project/Flickr-YsR/merlionImages/
# set dump_path = /mnt/windows/Project/DUMPS/offline/merlionImages/

python process_datasets.py $db_path $dump_path

