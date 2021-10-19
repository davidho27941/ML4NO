#!/usr/bin/bash


echo "study_test_sample_statistics"
date +"%r"
for i in {4500000..10000000..500000}  #10000000
do

    echo "$i"
    
    python3 study_test_sample_statistics.py $i  > ./Log/study_test_sample_statistics_"$i".log 
    

    
    
done





