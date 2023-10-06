#  3dVision_Rendering_Basics_with_Pytorch3D

## Where is the report/website is stored?

The report is stored in `docs/ starter.md.html`, additionally you can access the report online here https://darshit-desai.github.io/3dVision_Rendering_Basics_with_Pytorch3D/

## Code Running Instructions:

The code needs to be run using bash scripts which are in the root folder. After extracting the code from the zip file, open the root folder in the terminal where the bash scripts are stored. Make sure your envrionment has pytorch3d and it's deps installed.

All the code is housed in the starter folder, for running the python scripts, I have made bash files and everything can be run by running the bash files.

### Problem 0.1 code running instructions:
Here's how to run the bash file for this python code:
    
    ./problem0_1.sh --image_size $image_size #image_size is a number like 256 or 512

The bash command for regenerating the output of my report:

    ./problem0_1.sh

Output is stored in `images/cow_render.jpg`

### Problem 1.1 code running instructions:
Here's how to run the bash file for this python code:

    ##image_size could be 256/512; num_frames could be between 10 to 36; output_file is usually default image/360render_cow.gif    
    ./problem1_1.sh --num_frames "$NUM_FRAMES" --duration "$DURATION" --output_file "$OUTPUT_FILE" --image_size "$IMAGE_SIZE"

The bash command for regenerating the output of my report:

    ./problem1_1.sh --num_frames 36

The output is stored in `images\360render_cow.gif`.

### Problem 1.2 code running instructions:
Here's how to run the bash file for this python code:

    #Duration set as 3 seconds can be changed, No. of frames can be changed, output file kept default as images/dolly.gif; image_size kept as 256
    ./problem1_2.sh --num_frames "$NUM_FRAMES" --duration "$DURATION" --output_file "$OUTPUT_FILE" --image_size "$IMAGE_SIZE"

The bash command for regenerating the output of my report:

    ./problem1_2.sh --num_frames 10

The output is stored in `images\dolly.gif`.

### Problem 2.1 and 2.2 code running instructions:
This bash script generates output for both the tasks of Tetrahedron ad cube 360 degree rendering.

    ./problem2.sh

The output is stored at path `images/360render_tetrahedron.gif` and `images/cube_360.gif` for tetrahedron and cube respectively

### Problem 3 code running instructions:
The bash script for regenerating the outputs of my report is given below:

    ./problem3.sh --num_frames 36 --image_size 512

The output is stored at path `images/retexture_360render.gif`

### Problem 4 code running instructions:
The bash script for getting all 4 views as generated and shown in the report is given below:

    ./problem4.sh

The output is stored at the following paths:

* images/transform_cow1.jpg
* images/transform_cow2.jpg
* images/transform_cow3.jpg
* images/transform_cow4.jpg

### Problem 5 code running instructions:

#### Problem 5.1 code running instructions:
The bash script for getting the generated outputs as shown in reports are as below:

    ./problem5_1.sh

The outputs of the 3 point cloud renders are stored at the following path:

* images/ PC1_360render.gif
* images/ PC2_360render.gif
* images/ PCUnion_360render.gif

#### Problem 5.2 code running instructions:
The bash script for getting the generated outputs as shown in the reports are as below:

    ./problem5_2.sh --num_samples 300 --render parametric

The output for the script is stored at the path `images/toroid_parametric_360render.gif`

#### Problem 5.3 code running instructions:
The bash script for getting the generated outputs as shown in the reports are as below:

    ./problem5_3.sh --render implicit

The output for the script is stored at the path `images/toroid_parametric_360render.gif`



