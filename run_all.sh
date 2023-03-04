#!/bin/bash

# Define the source and target datasets
datasets=("mnist" "fashion" "cifar10" "cifar100" "svhn")

# Loop over each source dataset
for source in "${datasets[@]}"
do
    # Define the number of classes for the current source dataset
    if [ "$source" = "cifar100" ]
    then
        num_classes=20
    else
        num_classes=10
    fi

    # Loop over each class in the current source dataset
    for ((class=0; class<$num_classes; class++))
    do
        # Loop over each target dataset
        for target in "${datasets[@]}"
        do
            echo "Running fid_score.py with source dataset=$source, source class=$class, and target dataset=$target"
            python fid_score.py \
            --source_dataset $source \
            --source_class $class \
            --exposure_dataset $target \
            --device cuda:0 \
            --batch-size 256    
        done
    done
done
