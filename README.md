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
## Future Steps
* Add supprot for 3-D images (RGB)
* Add more layers (MAX pooling, Batch Normalization, flatten, Dense, Activation etc.)
* Create proper pipeline i.e., output of 1 layers becomes input to the next layer.
* Create some activation functions (Softmax, Sigmoid, Relu etc.)
* Study weights file and how to parse it.
* Make a small classification CNN for testing. (weights loading issue will come here ;-/ )
* Study yolo to see which layers, activations etc. it requires
* Implement simple yolo model in mojo
* Study yolo weights file and load yolo weights in my model.
## Issues which might
* n-dimentional array stacks in CNN
* Major problem: How to load the weights into my model? Parsing the weights file.
