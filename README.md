# Mojo-Yolo
# Current Status:
## 1. Data Structures
Implemented the data structure to store images and filters (right now only support gray scale images i.e., 2-D array)
## 2. Image Processing
Implemented Image reading, writing, preprocessing etc.
## 3. Layers
Implementation of CNN layers
### - Conv2d
Implemented simple convolution layer for gray scale images
## 4. Examples
Vertical egde detection filter output on example image by passing it through conv2D layer
## Future Steps
* Add supprot for 3-D images (RGB)
* Add more layers (MAX pooling, Batch Normalization, flatten, Dense, Activation etc.)
* Create proper pipeline i.e., output of 1 layers becomes input to the next layer.
* Create some activation functions (Softmax, Sigmoid, Relu etc.)
* Study weights file and how to parse it.
* Make a small classification CNN for testing. (weights loading issue will come here ;-/ )
* Study yolo to see which layers, activations etc. it requires
* Implement simple yolo model in mojo
* Implement additional functions required for yolo (IOU etc.)
* Study yolo weights file and load yolo weights in my model.
## Issues
* n-dimentional array stacks in CNN
* Major problem: How to load the weights into my model? Parsing the weights file.
