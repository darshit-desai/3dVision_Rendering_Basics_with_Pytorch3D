#!/bin/bash

# Command-line arguments
NUM_FRAMES=10
DURATION=3
OUTPUT_FILE="images/retexture_360render.gif"
IMAGE_SIZE=256

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
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

# Run the Python script
python -m starter.cow_retexture --num_frames "$NUM_FRAMES" --duration "$DURATION" --output_file "$OUTPUT_FILE" --image_size "$IMAGE_SIZE"
