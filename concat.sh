#!/bin/bash

# # fix all annotations
# python3 scripts/fix_annotation.py \
#     --annotation_path "datasets/pothole_2/train_filtered_0-1000/annotations.json" \
#     --output_path "datasets/pothole_2/train_filtered_0-1000/annotations_fixed.json" 

# echo "Done 0-1000"

# concatenate annotations
python3 scripts/concatenate.py \
    --first_annotation_path "datasets/pothole_0.json" \
    --second_annotation_path "datasets/pothole_1.json" \
    --new_annotation_path "datasets/pothole_0-1.json"

python3 scripts/concatenate.py \
    --first_annotation_path "datasets/pothole_0-1.json" \
    --second_annotation_path "datasets/pothole_2.json" \
    --new_annotation_path "datasets/pothole_full.json"

echo "Done!"
