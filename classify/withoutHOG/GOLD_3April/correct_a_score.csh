#!/bin/csh -f

if($#argv != 1 ) then
    echo "pass dataset path"
    exit()
endif

set file_name = $argv[1]
set new_file = "${file_name}.corrected"
unlink $new_file
set lines = `cat $file_name`

set i = 1

while($i < $#lines)
    set a_score = $lines[$i]
    set new_score = `echo $a_score | awk '{print ($1 > 0.78)?1:0}'`
    @ i += 1
    set idx = `echo $i | awk '{print $1 + 251}'`
    set new_line = `echo $lines[$i-$idx]`
    echo $new_score $new_line >> $new_file
    @ i = ($idx + 1)

end

