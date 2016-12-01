#!/bin/csh 

set db_path = "/mnt/windows/DataSet-YsR/"
set dump_path = "/home/vyzuer/DUMPS/offline/"

foreach location (arcde colognecathedral gatewayofindia indiagate leaningtower liberty merlion taj vatican eifel esplanade floatMarina)

    echo $location
    set dst_dir = "${dump_path}${location}/params/"
    mkdir -p $dst_dir
    foreach f_name (".comp.params" ".ap.params" ".ev.params" ".ss.params")
        set f_src =  "${db_path}${location}/${f_name}"
        set f_dst =  "${dst_dir}${f_name}"

        cp $f_src $f_dst
    end
end
