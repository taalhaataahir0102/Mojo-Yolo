# YOLO Implementation in Mojo

## Introduction

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. Unlike traditional object detectors that scan an image at multiple scales and locations, YOLO frames object detection as a single regression problem, directly providing class probabilities and bounding box coordinates. This innovative approach allows YOLO to achieve remarkable speed and accuracy in object detection tasks.

## Project Overview

This project implements the YOLO object detection algorithm using Mojo, incorporating several fundamental layers necessary for neural networks:

- **Convolutional Layers**
- **Max Pooling Layers**
- **Dense (Fully Connected) Layers**
- **Batch Normalization Layers**
- **Activation functions**

Other than mojo we've also run yolo inference via pytorch and max engine and compared the performance 
## Performance

1. For smaller models, Mojo outperforms PyTorch and MAX Engine by 1.3x.
2. As model size increases, Mojo's performance decreases, and at 250K parameters, PyTorch surpasses Mojo.
3. Pytorch and MAX engine are performing almost equivalent.

![Screenshot](https://raw.githubusercontent.com/taalhaataahir0102/Mojo-Yolo/main/Mojo/Example/Screenshot%20from%202024-07-24%2013-47-13.png)


Profiling revealed that the matmul function in Mojo is significantly slower, consuming most computation time during inference. As model weight matrices grow, performance drops. For profiling my Mojo code, I used the time library available in Mojo and measured the time taken by different functions.

## Execution

To run mojo inference, go inside the Mojo/main.mojo and update the yolo_model function and add the weigts file in Mojo/Parser (.h5 file format) and run the main.mojo file

For pytorch simply run the run.py file  and for MAX execute run.mojo file

## Future Work

1. Improve the matmul function by exposing L2 and L3 caching.
2. Enhance Mojo data structures by introducing efficient slicing.
3. Utilize MAX Engine for matmul, leveraging their optimized kernels for various CPUs.
