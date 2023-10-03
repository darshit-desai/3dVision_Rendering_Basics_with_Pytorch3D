#!/bin/bash

# Default values
IMAGE_SIZE=256
NUM_SAMPLES=100

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --render)
            RENDER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Run the Python script with default render value "point_cloud"
python -m starter.render_toroid --image_size "$IMAGE_SIZE" --num_samples "$NUM_SAMPLES" --render parametric
