#!/bin/csh -f

set db_path = /home/vyzuer/DUMPS/eigen_analysis/DB_DB1/
set model_path = /home/vyzuer/DUMPS/eigen.1/DB1/

python popular_composition.py ${db_path} ${model_path}

