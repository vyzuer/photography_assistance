#!/bin/csh -f

if($#argv != 4 ) then
    echo "Pass 0: Dump path 1: data file 2: target file 3: aesthetic scores"
        exit()
    endif

set mark = 0.30

set dump_path = $argv[1]
set data = $argv[2]
set target = $argv[3]
set a_score = $argv[4]

set lines = `cat $data`
set score = `cat $a_score`

set cam_param = "${dump_path}/cam.settings"
set map = "${dump_path}/map.list"
set fv = "${dump_path}/fv.list"

unlink $map
unlink $cam_param
unlink $fv

set j = 1
set i = 1

foreach l ("`cat $target`")
    set idx = `echo $i | awk '{print $1 + 13}'`
    set temp = `echo $l | grep null`
    set flag = 1
    set good_pic = `echo $score[$j] $mark | awk '{ print ($1 > $2) ? 1 : 0}'`
    echo $score[$j]
    echo $good_pic
    if("xxx" == "xxx$temp" && $good_pic == 1) then
        echo $l >> $cam_param
        set info = `echo $lines[$i-$idx]`
        echo $info >> $fv
        set flag = 0
    endif
    echo $flag >> $map
    @ i = ($idx + 1)
    @ j = ($j + 1)
end

