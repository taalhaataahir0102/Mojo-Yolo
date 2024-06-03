from python import Python
from tensor import Tensor, TensorShape
from parser.weights import collect_weights_2
from Imageprocessing.read import read_image_mask
from Imageprocessing.save import save_image_mask
from Layers.conv import Conv2Dim2col
from Layers.dense import dense
from Layers.maxpool import maxpool
from utils.index import Index
from time import now
import pathlib
import os
from python import Python as py


fn Tensor_to_numpy(img: Tensor[DType.float32]) raises -> PythonObject:
    var np = Python.import_module("numpy") 
    var l:Int = img.shape()[0] * img.shape()[1]
    var numpy_array = np.zeros(l)
    for i in range(l):
        numpy_array[i] = img[i]
    return numpy_array

def main():

    var time_in_matmul:Float32 = 0.0

    var conv2D_weights = List[Tensor[DType.float32]] ()
    var dense_weight = List[Tensor[DType.float32]] ()
    var baises = List[Tensor[DType.float32]]()

    collect_weights_2(conv2D_weights, dense_weight, baises)

    var image_path = "Example/mask.jpeg"

    var image = read_image_mask(image_path)


    var tens = Tensor[DType.float32] (image.shape[0].__int__(),image.shape[1].__int__(),image.shape[2].__int__(), 1)

    save_image_mask(tens, image)

    print("tens.shape: ",tens.shape())

    var layer1  = Conv2Dim2col(tens, conv2D_weights[0], baises[1])
    print("layer1.shape: ",layer1.shape())

    var layer2  = maxpool(layer1,2, 2)
    print("layer2.shape: ",layer2.shape())

    var layer3  = maxpool(layer2,2, 2)
    print("layer3.shape: ",layer3.shape())

    var layer4  = maxpool(layer3,2, 2)
    print("layer4.shape: ",layer4.shape())

    var layer5  = Conv2Dim2col(layer4, conv2D_weights[1], baises[2])
    print("layer5.shape: ",layer5.shape())

    var out_shape = TensorShape(1, layer5.shape()[0] * layer5.shape()[1] *layer5.shape()[2] *layer5.shape()[3])
    var layer6 = layer5.reshape(out_shape)
    print("layer6.shape: ",layer6.shape())

    var layer7 = dense(layer6, dense_weight[1], baises[3],"relu")
    print("layer7.shape: ",layer7.shape())

    var regressor_head = dense(layer7, dense_weight[2], baises[4], "")
    print("regressor_head:", regressor_head.shape())

    var classifier_head = dense(layer7, dense_weight[0], baises[0], "sigmoid")
    print("classifier_head:", classifier_head.shape())

    var predicted_label:PythonObject = Tensor_to_numpy(classifier_head)
    var predicted_box:PythonObject = Tensor_to_numpy(regressor_head)

    print(predicted_label, predicted_box)
    
    var input = py.import_module("builtins").input
    Python.add_to_path("/home/talha/Desktop/mojo/tensor_yolo")
    var mypython = Python.import_module("draw")
    var dict: PythonObject = mypython.process_and_predict("Example/mask.jpeg", 100, predicted_label, predicted_box)