#!/usr/bin/bash




for i in {0..10..1}
do

    nohup python3 dropout_study.py dune theta23 $i > dune_theta23_"$i"_standardized.log &
    
#     nohup python3 dropout_study.py dune delta $i > dune_delta_"$i"_standardized.log &

done