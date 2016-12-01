#!/bin/csh -fv

set a_score = "aesthetic.scores"
set feature_list_tr = "features.tr"
set feature_list_te = "features.te"

set lines = `cat $a_score`

cut -d' ' -f 2-253 $feature_list_tr > _temp1
cut -d' ' -f 2-253 $feature_list_te > _temp2

set f1 = "f1.tr"
set f2 = "f1.te"
unlink $f1
unlink $f2

set lines1 = `cat _temp1`

set len1 = `wc _temp1 | cut -d' ' -f 6`

set i = 1
set j = 1
set jx = 1
while($i < $#lines)

    if($jx >= $len1) then
        set lines1 = `cat _temp2`
        set j = 1
        set f1 = "f1.te"
    endif

    set jx = `echo $j | awk '{print $1+251}'`
    
    set score = `echo $lines[$i]`
    set fv = `echo $lines1[$j-$jx]`
    echo $score $fv >> $f1

    @ i += 1
    @ j = ($jx + 1)

end

#rm -rf _temp1 _temp2
