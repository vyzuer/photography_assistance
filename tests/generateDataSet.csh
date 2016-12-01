#!/bin/csh -f

if($#argv != 2 ) then
    echo "pass dataset path"
    exit()
endif

set dbPath = $argv[1] 
set dump_path = $argv[2]

set image_list = ${dbPath}/image.list
set a_score = ${dbPath}/aesthetic.scores

set lines = `cat $image_list`
set scores = `cat $a_score`
set num_sample = `wc -l $image_list`

set train_file = ${dump_path}/features.tr
set test_file = ${dump_path}/features.te

unlink $train_file
unlink $test_file

set num_train = `echo $num_sample | awk '{print int(0.8*$1)}'`

set dump_file = $train_file

set i = 1

while($i < $#lines)
    set image_src = ${dbPath}/ImageDB/$lines[$i]
    python compile.py $dump_path $image_src $scores[$i] $dump_file
    @ i += 1
    if( $i > $num_train ) then
        set dump_file = $test_file
    endif

end

