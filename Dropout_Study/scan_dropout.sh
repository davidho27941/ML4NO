#!/usr/bin/bash


# echo "dropout_study"
# date +"%r"
# for i in {0..4..1}
# do

#     nohup python3 dropout_study.py dune theta23 $i > ./Log/dune_theta23_"$i"_dropout.log &
    
#     nohup python3 dropout_study.py dune delta $i > ./Log/dune_delta_"$i"_dropout.log &

# done



# sleep 1h

# echo "dropout_study"
# date +"%r"
# for i in {5..9..1}
# do

#     nohup python3 dropout_study.py dune theta23 $i > ./Log/dune_theta23_"$i"_dropout.log &
    
#     nohup python3 dropout_study.py dune delta $i > ./Log/dune_delta_"$i"_dropout.log &

# done


# sleep 1h



# echo "dropout_study_standardized"
# date +"%r"
# for i in {0..4..1}
# do

#     nohup python3 dropout_study_standardized.py dune theta23 $i > ./Log/dune_theta23_"$i"_dropout_standardized.log &
    
#     nohup python3 dropout_study_standardized.py dune delta $i > ./Log/dune_delta_"$i"_dropout_standardized.log &

# done



# sleep 1h

echo "dropout_study_standardized"
date +"%r"
for i in {5..9..1}
do

    nohup python3 dropout_study_standardized.py dune theta23 $i > ./Log/dune_theta23_"$i"_dropout_standardized.log &
    
    nohup python3 dropout_study_standardized.py dune delta $i > ./Log/dune_delta_"$i"_dropout_standardized.log &

done




