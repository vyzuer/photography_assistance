#!/bin/csh -f

set db_path = /home/vyzuer/DUMPS/offline.1/

# foreach location (arcde)
foreach location (arcde colognecathedral gatewayofindia indiagate leaningtower liberty merlion taj vatican esplanade floatMarina eifel)

    echo $location
    python composition.py "${db_path}${location}/" 
end
