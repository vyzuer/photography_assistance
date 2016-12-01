#!/bin/csh -f

set db_path = /home/vyzuer/DUMPS/eigen.1/
# set db_path = /home/vyzuer/DUMPS/offline.1/

foreach location (DB1 DB2)
# foreach location (arcde colognecathedral gatewayofindia indiagate leaningtower liberty merlion taj vatican esplanade floatMarina eifel)

    echo $location

    sed '/nan/d' "${db_path}${location}/feature_dumps/comp_map.list" > "${db_path}${location}/feature_dumps/comp_map_clean.list"
end
