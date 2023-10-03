#!/bin/bash

# Default values
COW_PATH="data/cow_with_axis.obj"
IMAGE_SIZE=256

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cow_path)
            COW_PATH="$2"
            shift 2
            ;;
        --image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Run the Python script without specifying output_path
python -m starter.camera_transforms --image_size "$IMAGE_SIZE" --cow_path "$COW_PATH"
