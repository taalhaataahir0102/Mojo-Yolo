from DataStructure.Array2D import Array2D, Array3D
from DataStructure.Array1D import Array1D
from ImageProcessing.read import read_image, read_image_yolo
from ImageProcessing.store import save_image
from ImageProcessing.write import write_image
from ImageProcessing.normalization import normalize
from Layers.Conv2D import Conv2D, pading
from Layers.Maxpooling import maxpool
from Layers.Flatten import flatten, flatten2, flatten3
from Layers.Dense import dense
from Activations.Relu import relu
from Activations.Sigmoid import sigmoid
# from Parser.weights import collect_weights, collect_conv2d_weights, collect_dense_weights, collect_baises
from Parser.mojo_weights import collect_weights
import math
from python import Python
from python import Python as py
from time import now

fn main() raises:

    var read_img_start = now()
    var image_path = "Example/carbird.jpeg"
    var image = read_image(image_path)

    print("image_shape length",len(image.shape))

    # var img = Array2D(30,30)
    # if len(image.shape) == 2:
    #     var img = Array2D(image.shape[0].__index__() , image.shape[1].__index__())
    # else:
    var img = Array3D(image.shape[2].__index__(),image.shape[0].__index__() , image.shape[1].__index__())

    save_image(img,image)
    # var new_img = Array3D(1,img.dim0, img.dim1)
    # for i in range(img.dim0):
    #     for j in range(img.dim1):
    #         new_img.__setitem__(0,i,j,img.__getitem__(i,j))
    print("Input image shape: ", img.dim0, img.dim1, img.dim2)
    print("IMAGE:")
    img.__printarray__()

    var read_img_end = now()
    var execution_time_read_img : Float32 = (read_img_end - read_img_start)
    var execution_time_seconds_read_img :  Float32 = execution_time_read_img / 1000000000


    var read_weights_start = now()
    var conv2D_weights = List[Array3D] ()
    var dense_weights = List[Array2D] ()
    var baises = List[List[Float32]]()

    collect_weights(conv2D_weights, dense_weights, baises)

    var read_weights_end = now()
    var execution_time_read_weights : Float32 = (read_weights_end - read_weights_start)
    var execution_time_seconds_read_weights :  Float32 = execution_time_read_weights / 1000000000


    #########OBJECT DETECTION YOLO MODEL##############

    var inference_start = now()


    var conv2d_1 = Conv2D(img,conv2D_weights[0], 1, 1, baises[0])
    print("Dimentions afterconv2d_1: ", conv2d_1.dim0,conv2d_1.dim1,conv2d_1.dim2)
    # conv2d_1.__printarray__()
    var maxpool1 = maxpool(conv2d_1,2)
    print("Dimentions after maxpool1: ", maxpool1.dim0,maxpool1.dim1,maxpool1.dim2)

    var conv2d_2 = Conv2D(maxpool1,conv2D_weights[1], 1, 1, baises[1])
    print("Dimentions after conv2d_2: ", conv2d_2.dim0,conv2d_2.dim1,conv2d_2.dim2)
    var maxpool2 = maxpool(conv2d_2,2)
    print("Dimentions after maxpool2: ", maxpool2.dim0,maxpool2.dim1,maxpool2.dim2)

    var conv2d_3 = Conv2D(maxpool2,conv2D_weights[2], 1, 1, baises[2])
    print("Dimentions after conv2d_3: ", conv2d_3.dim0,conv2d_3.dim1,conv2d_3.dim2)
    var maxpool3 = maxpool(conv2d_3,2)
    print("Dimentions after maxpool3: ", maxpool3.dim0,maxpool3.dim1,maxpool3.dim2)

    var conv2d_4 = Conv2D(maxpool3,conv2D_weights[3], 1, 1, baises[3])
    print("Dimentions after conv2d_4: ", conv2d_4.dim0,conv2d_4.dim1,conv2d_4.dim2)
    var maxpool4 = maxpool(conv2d_4,2)
    print("Dimentions after maxpool4: ", maxpool4.dim0,maxpool4.dim1,maxpool4.dim2)

    var conv2d_5 = Conv2D(maxpool4,conv2D_weights[4], 1, 1, baises[4])
    print("Dimentions after conv2d_5: ", conv2d_5.dim0,conv2d_5.dim1,conv2d_5.dim2)
    var maxpool5 = maxpool(conv2d_5,2)
    print("Dimentions after maxpool5: ", maxpool5.dim0,maxpool5.dim1,maxpool5.dim2)


    # var inference_start = now()
        
    var flat = flatten3(maxpool5)
    print("Dimentions after flatten: ", flat.dim0, flat.dim1)

    var dense_1 = dense(flat,dense_weights[0], baises[5])
    var dense_1_act = relu(dense_1)
    print("Dimentions after dense1: ", dense_1_act.dim0, dense_1_act.dim1)
    # dense_1_act.__printarray__()

    var dense_2 = dense(dense_1_act,dense_weights[1], baises[6])
    var dense_2_act = relu(dense_2)
    print("Dimentions after dense1: ", dense_2_act.dim0, dense_2_act.dim1)

    var dense_3 = dense(dense_2_act,dense_weights[2], baises[7])
    var dense_3_act = sigmoid(dense_3)
    print("Dimentions after dense1: ", dense_3_act.dim0, dense_3_act.dim1)
    dense_3_act.__printarray__()
    

    var inference_end = now()
    var execution_time_inference : Float32 = (inference_end - inference_start)
    var execution_time_seconds_inference :  Float32 = execution_time_inference / 1000000000

    dense_3_act.save_to_file("/home/lpt-10x/Desktop/Mojo-Yolo/final_layer.txt")

    print("Time taken to read and pre-process image:", execution_time_seconds_read_img)
    print("Time taken to parse weights file:       ", execution_time_seconds_read_weights)
    print("Time take to run yolo inference model:  ", execution_time_seconds_inference)