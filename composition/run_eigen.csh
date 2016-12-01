#!/bin/csh -f

set db_path = /home/vyzuer/DUMPS/eigen.1/

foreach location (DB2 DB1)
    echo $location
    python composition.py "${db_path}${location}/" 
end
