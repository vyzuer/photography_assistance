#!/bin/csh 


# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /home/vyzuer/DUMPS/eigen_analysis/
python xtract_composition.py ${db_path} 

# foreach location (arcde colognecathedral eifel)
# foreach location (gatewayofindia indiagate leaningtower liberty)
# foreach location (taj)
#     echo $location
#     python xtract_composition.py ${db_path}$location
# end

