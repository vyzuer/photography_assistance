#!/bin/csh -f

set dataset_path = /mnt/windows/DataSet-YsR/
set dump_path = /home/vyzuer/DUMPS/test_camera_motion/

set img_details = "images.details"
set img_list = "image.list"
set geo_list = "geo.info"
set weather_list = "weather.info"
set cam_list = "camera.settings"
set a_list = "aesthetic.scores"

foreach location (`ls $dump_path`)
# foreach location (arcde colognecathedral esplanade floatMarina gatewayofindia eifel indiagate leaningtower liberty merlion taj vatican)
    echo $location
    set db_path = "${dataset_path}${location}/"
    set target_path = "${dump_path}${location}/"
    set img_db = "${target_path}ImageDB/"

    mkdir -p $target_path
    mkdir -p $img_db

    set d_img_details = `cat ${db_path}${img_details}`
    set d_img_list = `cat ${db_path}${img_list}`
    set d_geo_list = `cat ${db_path}${geo_list}`
    set d_weather_info = `cat ${db_path}${weather_list}`
    set d_cam_list = `cat ${db_path}${cam_list}`
    set d_a_list = `cat ${db_path}${a_list}`

    echo $d_img_details[1-13] > ${target_path}${img_details}

    touch ${target_path}${img_list}
    unlink ${target_path}${img_list}

    touch ${target_path}${a_list}
    unlink ${target_path}${a_list}

    touch ${target_path}${cam_list}
    unlink ${target_path}${cam_list}

    touch ${target_path}${weather_list}
    unlink ${target_path}${weather_list}

    touch ${target_path}${geo_list}
    unlink ${target_path}${geo_list}

    foreach img (`ls $img_db`)
        echo $img
        set l_num = `grep -n $img ${db_path}${img_details} | cut -d":" -f 1`
        echo $l_num
        if ( "xxx$l_num" == "xxx") then
            continue
        endif

        set i = `echo $l_num | awk '{print ($0 - 2)*5 + 1}'`
        set idx = `echo $i | awk '{print $0 + 4}'`
        set null_chk = `echo $d_cam_list[$i-$idx] | grep null`
        if ("xxx$null_chk" != "xxx") then
            continue
        endif

        echo $d_cam_list[$i-$idx] >> ${target_path}${cam_list}

        set i = `echo $l_num | awk '{print ($0 -1)*13 + 1}'`
        set idx = `echo $i | awk '{print $0 + 12}'`
        echo $d_img_details[$i-$idx] >> ${target_path}${img_details}

        echo $img >> ${target_path}${img_list}

        set i = `echo $l_num | awk '{print ($0 - 2)*13 + 1}'`
        set idx = `echo $i | awk '{print $0 + 12}'`
        echo $d_weather_info[$i-$idx] >> ${target_path}${weather_list}

        set i = `echo $l_num | awk '{print $0-1}'`
        echo $d_a_list[$i] >> ${target_path}${a_list}

        set i = `echo $l_num | awk '{print ($0 - 2)*2 + 1}'`
        set idx = `echo $i | awk '{print $0 + 1}'`
        echo $d_geo_list[$i-$idx] >> ${target_path}${geo_list}
        
    end
end

