#!/bin/csh 

# set dataset_path = /mnt/windows/Project/InterestingDB/sample_images/
set dataset_path = /home/vyzuer/DUMPS/eigen_analysis/eigen_rules.2/
# set dataset_path = /mnt/windows/Project/Flickr-code/pos_results/comp/
set dump_path = /home/vyzuer/DUMPS/eigen_analysis/eigen_rules.2/DUMP/

python process_datasets_eigen.py "${dataset_path}" "${dump_path}"

