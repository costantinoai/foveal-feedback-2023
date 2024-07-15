#!/bin/zsh

# Define file mappings as an associative array
declare -A file_mappings=(
  ["~/Dropbox/andrea/rois/sub-02/sub-02_space-MNI152NLin2009cAsym_label-A1_roi.nii"]="data/BIDS/derivatives/rois/sub-02/sub-02_space-MNI152NLin2009cAsym_label-A1_roi.nii"
)

# Loop through the file mappings and copy the files
for source_name in "${(@k)file_mappings}"; do
  target_name="${file_mappings[$source_name]}"
  cp "$source_name" "$target_name"
  echo "Copied $source_name to $target_name"
done
