#!/bin/csh -f

if($#argv != 1 ) then
    echo "Pass 0: database"
        exit()
    endif

set db_path = $argv[1]

set f_param = "${db_path}/camera.settings"

set map = "${db_path}/map.list"

unlink $map

set i = 1

foreach l ("`cat $f_param`")
    set temp = `echo $l | grep null`
    set flash = `echo $l | cut -d" " -f 5`
    set ss_valid = `echo $l | cut -d" " -f 1 | awk '{print ($1 > 1) ? 0 : 1}'`
    set flag = 0
    if("xxx" == "xxx$temp" && $flash == 0 && $ss_valid == 1) then
        set flag = 1
    endif
    echo $flag >> $map
end

