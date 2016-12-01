#!/bin/csh -f

set db_path = /home/yogesh/Project/Flickr-YsR/InterestingDB/


foreach location (${db_path}/*)

    echo $location

    python composition.py "${location}/" "/"
end
