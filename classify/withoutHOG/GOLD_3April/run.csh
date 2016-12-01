#!/bin/csh 

./correct_a_score.csh features.tr
./correct_a_score.csh features.te

cut -d' ' -f1-181 features.te.corrected > fv.te
cut -d' ' -f1-181 features.tr.corrected > fv.tr

python compile.py

