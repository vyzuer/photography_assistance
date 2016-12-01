#!/bin/csh -f

# set dataset_path = "/home/yogesh/Project/Flickr-YsR/merlionImages/Good_Image_DB/segment_dump/"
set dataset_path = "/home/vyzuer/Copy/Flickr-code/PhotographyAssistance/testing/DB2/"
# set dump_path = "/home/yogesh/Copy/Flickr-code/DBR/VC-27May/"
set dump_path = "/home/yogesh/Project/Flickr-YsR/DBR/VC-16_1/"
set file_name = "${dump_path}segments.list"

echo "Merging segmented visual words for clustering..."
./mergeVisualWords.csh $dataset_path $dump_path

set num_clusters = 50
echo "Cluster visual words..."
python test.py $dump_path $file_name $num_clusters

echo "creating clusters..."
./createSegmentClusters.csh $dump_path 

