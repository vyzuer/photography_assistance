#!/bin/csh -f

# set db_path = /home/vyzuer/DUMPS/eigen_analysis/sample_images/
# set db_path = /home/vyzuer/DUMPS/eigen_analysis/eigen_rules/DUMP/
set db_path = /home/vyzuer/DUMPS/eigen_analysis/eigen_rules.2/DUMP/
set model_path = /home/vyzuer/DUMPS/eigen.1/DB1/
# set model_path = /home/vyzuer/DUMPS/offline.1/merlion/

python eigen_analysis.py ${db_path} ${model_path}

