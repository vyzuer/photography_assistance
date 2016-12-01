#!/bin/csh 

set db_path = "/mnt/windows/Project/DUMPS/offline/"
set dump_path = "/home/vyzuer/DUMPS/offline/"

foreach location (arcde colognecathedral gatewayofindia indiagate leaningtower liberty merlion taj vatican eifel esplanade floatMarina)

    echo $location
    set dst_dir = "${dump_path}${location}/cluster_dump/"

    mkdir -p $dst_dir
    pushd $dst_dir

    foreach f_name ("cluster_model" "SegClustersInfo" "popularity.score")
        set f_src =  "${db_path}${location}/cluster_dump/${f_name}"

        cp -rf $f_src .
    end

    popd

#     set f_name = "camera.settings"
#     set dst_dir = "${dump_path}${location}/"
#     set f_src =  "${db_path}${location}/${f_name}"
#     set f_dst =  "${dst_dir}${f_name}"
# 
#     cp $f_src $f_dst
end
