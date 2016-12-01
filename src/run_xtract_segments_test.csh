#!/bin/csh 


set db_path = /home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/
set dump_path = /home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/dump/

python xtract_segments.py $db_path $dump_path

