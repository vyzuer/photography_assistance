#!/bin/csh -fv

set cam = "camera.settings"
set fv = "features_cp.list"

set lines = `cat $fv`

set new_ffile = "fv.list"
set new_cam_file = "settings.list"
set map = "map.list"
unlink $map
unlink $new_ffile
unlink $new_cam_file

set i = 1

foreach l ("`cat $cam`")
    set idx = `echo $i | awk '{print $1 + 20}'`
    set temp = `echo $l | grep null`
    set flag = 1
    if("xxx" == "xxx$temp") then
        echo $l >> $new_cam_file
        set info = `echo $lines[$i-$idx]`
        echo $info >> $new_ffile
        set flag = 0
    endif
    echo $flag >> $map
    @ i = ($idx + 1)
end

