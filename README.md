# VPoser C++

This repository contains a working C++ implementation using libtorch for the VPoser.  
It only contains the actual VPoser itself for inference without the training and stuff.  
The implementation is as close as possible to the original.  
For the official python implementation please visit [VPoser](https://github.com/nghorbani/human_body_prior).


## Setup

1. You need a current g++, cmake and anaconda or miniconda.

2. Setup the environment: 
You can use the provided `setup_env.sh`.  
This will create two conda environments; one for step 3 and one in which libtorch is stored.  
The script assumes that you anaconda is installed at `~/anaconda3`, if that is not the case adjust `CONDA_PATH` inside the script.

3. Download the VPoser v2.0 model files from [here](https://smpl-x.is.tue.mpg.de/) and unpack the zip, note that the model files are subject to their license.  
These files must be converted using the `vposer_convert.py` due, because they are not directly compatible with C++.  
Usage:  
```
//First activate environment
conda activate vposer_conv
//Actual conversion with path to the yaml file inside the unzipped folder
python vposer_convert.py -p "./EXAMPLE_PATH/V02_05.yaml"
```
 
## Build
CMake adjust the `PREFIX_PATH` if you anaconda environment is saved somewhere else and activate conda environment `vposer`, this removes a warning in cmake.

Build VPoser library only:
```
cmake -S . -B build -Wno-dev -DCMAKE_PREFIX_PATH="~/anaconda3/envs/vposer/;~/anaconda3/envs/vposer/lib/python3.10/site-packages/torch/" && cmake --build build --parallel
```
Build with example:
```
cmake -S . -B build -Wno-dev -DCMAKE_PREFIX_PATH="~/anaconda3/envs/vposer/;~/anaconda3/envs/vposer/lib/python3.10/site-packages/torch/" -DVPOSER_BUILD_EXAMPLES=ON && cmake --build build --parallel
```
You also need to set the path to the npz file in `example.cpp`.
   
Build without cuda:
```
cmake -S . -B build -Wno-dev -DCMAKE_PREFIX_PATH="~/anaconda3/envs/vposer/;~/anaconda3/envs/vposer/lib/python3.10/site-packages/torch/" -DBUILD_WITH_CUDA=OFF && cmake --build build --parallel
```
