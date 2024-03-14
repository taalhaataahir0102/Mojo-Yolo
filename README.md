# Mojo-Yolo
Implementation of yolo inference model in mojo
## Data Structures
Contains the data structure to store images and filters (right now only support gray scale images i.e., 2-D array)
## Image Processing
Image reading, writing, preprocessing etc.
## Layers
Implementation of CNN layers
### Conv2d
Implemented simple convolution layer for gray scale images
## Examples
Vertical egde detection filter output on example image
## Future Steps before next sync-up
* Add supprot for 3-D images (RGB)
* Add more layers (MAX pooling, Batch Normalization, flatten, Dense etc.)
* Create proper pipeline i.e., output of 1 layers becomes input to the next layer.
* Make a small classification CNN for testing. (weights loading issue will come here ;-/ )
* Study weights file and how to parse it.
## Issues
* n-dimentional array stacks in CNN
* Major problem: How to load the weights into my model?
