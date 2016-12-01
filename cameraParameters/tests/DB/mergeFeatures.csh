#!/bin/csh -fv

if($#argv != 2 ) then
    echo "two files to merge"
    exit()
endif

set f1 = $argv[1]
set f2 = $argv[2]

set lines = `cat $f1`

set new_file = features_cp.list
unlink $new_file

set i = 1

foreach l ("`cat $f2`")
    echo $lines[$i] $l >> $new_file
    @ i = ($i + 1)
end

