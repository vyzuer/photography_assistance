#!/bin/csh -f

# set db = "/home/yogesh/Project/Flickr-YsR/"
# set location = "/esplanade/"
# set location = "/floatMarina/"
# set location = "/merlionImages/"
# set dump_dir = "/dump_ImageDB_640/"

# set db = "/mnt/windows/DataSet-YsR/"
set db = "/home/vyzuer/DUMPS/offline/"
# set location = "/arcde/"
set dump_dir = "/feature_dumps/"

# foreach location (${db}/*)
# foreach location (`ls ${db}/`)
foreach location (arcde colognecathedral gatewayofindia indiagate leaningtower liberty merlion vatican esplanade floatMarina eifel)
# foreach location (arcde)

    set db_path = "${db}/${location}/"
    
#     echo "filtering bad features..."
#     ./filter_features.csh $db_path 
#     echo "done."
#     
#     echo "evaluate ev score..."
#     python filter_features.py $db_path $dump_dir
#     echo "done."

    echo "learning model for camera parameters..."
    
    python classify.py $db_path $dump_dir
    python classify_ap.py $db_path $dump_dir
    python classify_ss.py $db_path $dump_dir
    
    echo "done."

end    
