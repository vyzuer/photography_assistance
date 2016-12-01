#!/bin/csh 

# set db_path = "/mnt/windows/DataSet-YsR/"
set db_path = "/home/vyzuer/DUMPS/offline.1/"
set dump_dir = "/dump_ImageDB/"

# foreach location (`ls ${db_path}/ `)
foreach location (arcde colognecathedral gatewayofindia indiagate leaningtower liberty merlion taj vatican esplanade floatMarina eifel)
# foreach location (merlion)

    echo $location
    python bin_classify_dump.py "${db_path}/${location}/" $dump_dir

end
