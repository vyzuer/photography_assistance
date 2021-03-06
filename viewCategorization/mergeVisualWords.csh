#!/bin/tcsh -fv

# set db_path = "/home/scps/myDrive/Flickr-YsR/merlionImages/DB2April/"
# set dump_path = "/home/scps/myDrive/Copy/Flickr-code/DBR/clustering_1104/"

if($#argv != 2) then
    echo "Usage : $0 dataset_path dump_path"
    exit
endif

set db_path = $1
set dump_path = $2

mkdir -p $dump_path


# input

set details_file = "${db_path}/images.list"
set sal_file = "saliency.list"
set feature_file = "feature.list"


# OUTPUT
set seg_file = "${dump_path}segments.list"
set img_file = "${dump_path}image.list"
set sal_file2 = "${dump_path}${sal_file}"
set details_file_new = "${dump_path}/images.details"

pushd $dump_path  # [[ dbpath
mkdir -p SegDB

unlink $seg_file
unlink $img_file
unlink $sal_file2
unlink $details_file_new

set line_info = `cat $details_file`

set j = 13

while($j < $#line_info)
    echo $j
    set jx = `echo $j | awk '{print ($1+11)}'`
    echo $jx
    set info = `echo $line_info[$j-$jx]`

    set img_name = `echo "$line_info[$j]" | sed 's/\.jpg//'`
    set file = "${db_path}/${img_name}"
    echo $file
    if (-d $file) then
        # set img_name = `echo $file | sed 's:'"$db_path/"'::'`
        
        set s_file = "${file}/${sal_file}"
        set segments_path = "${file}/segments/"
        set f_file = "${file}/${feature_file}"
        
        set num_seg = `wc -l $f_file | cut -d' ' -f 1`
        
        echo "${img_name} ${num_seg}"  >> $img_file
        echo $info >> $details_file_new

        mkdir -p SegDB/${img_name}
        
        cat $f_file >> $seg_file
        head -n $num_seg $s_file >> $sal_file2

        set i = 1
        while($i <= $num_seg)
            set temp_file = "${segments_path}/${i}.png"
            ln -sf $temp_file SegDB/${img_name}/

            @ i += 1
        end

    endif
    @ j = ($jx + 1)
    echo $j
    echo $jx
end
popd

