#### Filter images
```
python main.py --images_path datasets/pothole_2/valid --annotation_path datasets/pothole_2/valid/annotations.json --output_path datasets/pothole_2/valid_filtered

python main.py --images_path datasets/pothole_0/all_filtered --annotation_path datasets/pothole_0.json --output_path datasets/t
```

#### Fix annotations
```
python3 scripts/fix_annotation.py \
    --annotation_path "datasets/pothole_1/train_filtered/annotations.json" \
    --output_path "datasets/pothole_1/train_filtered/annotations_fixed.json" 
```

#### Concatenate annotations
```
python3 scripts/concatenate.py \
    --first_annotation_path "datasets/pothole_1/train_filtered/annotations_fixed.json" \
    --second_annotation_path "datasets/pothole_1/valid_filtered/annotations_fixed.json" \
    --new_annotation_path "datasets/pothole_1_train_valid.json"
```

#### Prepare annotations
```
python scripts/prepare_annotations.py --annotation_path "datasets/pothole_full.json" --new_annotation_path "new.json"
```