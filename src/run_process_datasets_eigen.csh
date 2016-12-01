#!/bin/csh 

set dataset_path = /mnt/windows/Project/InterestingDB/
set dump_path = /home/vyzuer/DUMPS/eigen/

# foreach location (arcde colognecathedral esplanade floatMarina gatewayofindia eifel)
foreach location (DB1 DB2)
    echo $location
    python process_datasets_eigen.py "${dataset_path}${location}/" "${dump_path}${location}/"
end

