#!/bin/bash

# Command-line arguments for file 1
IMAGE_SIZE1=256
OUTPUT_FILE1="images/360render_tetrahedron.gif"

# Command-line arguments for file 2
IMAGE_SIZE2=256
OUTPUT_FILE2="images/cube_360.gif"

# Run the first Python script
python -m starter.tetrahedron_render --image_size "$IMAGE_SIZE1" --output_file "$OUTPUT_FILE1"

# Run the second Python script
python -m starter.cube_render --image_size "$IMAGE_SIZE2" --output_file "$OUTPUT_FILE2"
