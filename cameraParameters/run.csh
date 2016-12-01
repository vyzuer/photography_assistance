#!/bin/csh -f

set db = "/home/yogesh/Project/Flickr-YsR/"
# set location = "/esplanade/"
# set location = "/floatMarina/"
# set location = "/merlionImages/"
set dump_dir = "/dump_ImageDB_640/"

# set db = "/mnt/windows/DataSet-YsR/"
# set location = "/arcde/"
# set dump_dir = "/dump_ImageDB/"

# foreach location (${db}/*)
# foreach location (`ls ${db}/`)
foreach location (esplanade merlionImages floatMarina)

    set db_path = "${db}/${location}/"
    
    echo "filtering bad features..."
    ./filter_features.csh $db_path 
    echo "done."
    
    echo "evaluate ev score..."
    python filter_features.py $db_path $dump_dir
    echo "done."
    
    echo "learning model for camera parameters..."
    
    # python classify.py "./DB/fv.list" "./DB/luminance.score"
    python classify.py $db_path $dump_dir
    python classify_ap.py $db_path $dump_dir
    python classify_ss.py $db_path $dump_dir
    
    echo "done."

end    
