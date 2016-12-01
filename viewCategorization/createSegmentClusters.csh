#!/bin/csh -f

if($#argv != 1) then
    echo "Usage $0 data_set_path"
    exit
endif

set dbPath = $1

set labels_file = "labels.list"
set num_cluster_file = '_num_clusters.info'
set img_file = "image.list"

pushd dbPath # [[ dbPath

set num_of_clusters = `cat $num_cluster_file`
set file_list = `cat $img_file`
set clusterID = `cat $labels_file`

mkdir -p SegClusters
rm -rf SegClusters/*
pushd SegClusters  # [[ clusters

set i = 0
while ($i < $num_of_clusters)
    mkdir -p $i
    @ i++
end

set j = 1
set i = 1
while($j < $#clusterID)
    set img_name = $file_list[$i]
    @ i += 1
    set num_seg = $file_list[$i]
    
    set k = 1
    while($k <= $num_seg)
        set label = $clusterID[$j]
        if ( $label == -1 ) then
            @ k++
            @ j++
            continue
        endif
# echo $label
        @ j++
    
        set filename = "${dbPath}/SegDB/${img_name}/${k}.png"
        ln -sf $filename "${label}/${img_name}_${k}.png"
        @ k++
    end
    @ i++

end
# echo $num_of_clusters
popd  # ]] clusters

popd  # ]] dbPath

