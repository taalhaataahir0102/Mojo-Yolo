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

    
    var img = Array3D(image.shape[2].__index__(),image.shape[0].__index__() , image.shape[1].__index__())
    save_image(img,image)
    print("Input image shape: ", img.dim0, img.dim1,img.dim2)

    var read_img_end = now()
    var execution_time_read_img : Float32 = (read_img_end - read_img_start)
    var execution_time_seconds_read_img :  Float32 = execution_time_read_img / 1000000000


    var read_weights_start = now()
    var conv2D_weights = DynamicVector[Array3D] ()
    var dense_weights = DynamicVector[Array2D] ()
    var baises = DynamicVector[DynamicVector[Float32]]()

    collect_weights(conv2D_weights, dense_weights, baises)

    var read_weights_end = now()
    var execution_time_read_weights : Float32 = (read_weights_end - read_weights_start)
    var execution_time_seconds_read_weights :  Float32 = execution_time_read_weights / 1000000000

    print(len(baises), len(baises[0]))

    # print("IMAGE:")
    # img.__printarray__()

    print("Dimentions of image: ", img.dim0, img.dim1, img.dim2)

    #########OBJECT DETECTION YOLO MODEL##############

    var inference_start = now()


    var conv2d_1 = Conv2D(img,conv2D_weights[0], 1, 1, baises[0])
    print("Dimentions afterconv2d_1: ", conv2d_1.dim0,conv2d_1.dim1,conv2d_1.dim2)
    # conv2d_1.__printarray__()
    var maxpool1 = maxpool(conv2d_1,2)
    print("Dimentions after maxpool1: ", maxpool1.dim0,maxpool1.dim1,maxpool1.dim2)
    print("maxpool1:")
    # maxpool1.__printarray__()


    var conv2d_2 = Conv2D(maxpool1,conv2D_weights[1], 1, 1, baises[1])
    print("Dimentions after conv2d_2: ", conv2d_2.dim0,conv2d_2.dim1,conv2d_2.dim2)
    var conv2d_3 = Conv2D(conv2d_2,conv2D_weights[2], 1, 1, baises[2])
    print("Dimentions after conv2d_3: ", conv2d_3.dim0,conv2d_3.dim1,conv2d_3.dim2)
    var maxpool2 = maxpool(conv2d_3,2)
    print("Dimentions after maxpool2: ", maxpool2.dim0,maxpool2.dim1,maxpool2.dim2)
    print("maxpool2:")
    # maxpool2.__printarray__()

    var conv2d_4 = Conv2D(maxpool2,conv2D_weights[3], 1, 1, baises[3])
    print("Dimentions after conv2d_4: ", conv2d_4.dim0,conv2d_4.dim1,conv2d_4.dim2)
    var conv2d_5 = Conv2D(conv2d_4,conv2D_weights[4], 1, 1, baises[4])
    print("Dimentions after conv2d_5: ", conv2d_5.dim0,conv2d_5.dim1,conv2d_5.dim2)
    var maxpool3 = maxpool(conv2d_5,2)
    print("Dimentions after maxpool3: ", maxpool3.dim0,maxpool3.dim1,maxpool3.dim2)
    print("maxpool3:")
    # maxpool3.__printarray__()

    var conv2d_6 = Conv2D(maxpool3,conv2D_weights[5], 1, 1, baises[5])
    print("Dimentions after conv2d_6: ", conv2d_6.dim0,conv2d_6.dim1,conv2d_6.dim2)
    var maxpool4 = maxpool(conv2d_6,2)
    print("Dimentions after maxpool4: ", maxpool4.dim0,maxpool4.dim1,maxpool4.dim2)
    var conv2d_7 = Conv2D(maxpool4,conv2D_weights[6], 1, 1, baises[6])
    print("Dimentions after conv2d_7: ", conv2d_7.dim0,conv2d_7.dim1,conv2d_7.dim2)
    var maxpool6 = maxpool(conv2d_7,2)
    print("Dimentions after maxpool6: ", maxpool6.dim0,maxpool6.dim1,maxpool6.dim2)
    # maxpool6.__printarray__()



    var flat = flatten3(maxpool6)
    print("Dimentions after flatten: ", flat.dim0, flat.dim1)

    var dense_1 = dense(flat,dense_weights[0], baises[7])
    var dense_1_act = relu(dense_1)
    print("Dimentions after dense1: ", dense_1_act.dim0, dense_1_act.dim1)
    print("DENSE_1")
    # dense_1_act.__printarray__()

    var dense_2 = dense(dense_1_act,dense_weights[1], baises[8])
    var dense_2_act = relu(dense_2)
    print("Dimentions after dense2: ", dense_2_act.dim0, dense_2_act.dim1)
    print("DENSE_2")
    # dense_2_act.__printarray__()

    var dense_3 = dense(dense_2_act,dense_weights[2], baises[9])
    var dense_3_act = sigmoid(dense_3)
    print("Dimentions after dense3: ", dense_3_act.dim0, dense_3_act.dim1)
    print("DENSE_3")
    dense_3_act.__printarray__()

    var inference_end = now()
    var execution_time_inference : Float32 = (inference_end - inference_start)
    var execution_time_seconds_inference :  Float32 = execution_time_inference / 1000000000

    dense_3_act.save_to_file("/home/talha/Desktop/mojo/yolo/final_layer.txt")

    print("Time taken to read and pre-process image:", execution_time_seconds_read_img)
    print("Time taken to parse weights file:       ", execution_time_seconds_read_weights)
    print("Time take to run yolo inference model:  ", execution_time_seconds_inference)


    # print("DONE")
    # var input = py.import_module("builtins").input
    # Python.add_to_path("/home/talha/Desktop/mojo/yolo/")
    # print("DONE1")
    # var mypython = Python.import_module("run")
    # print("DONE2")
    # mypython.run()
    # print("DONE3")

    # var conv2d_1 = Conv2D(img,conv2D_weights[0], 1, 1, baises[0])
    # print("Dimentions after 1st conv: ", conv2d_1.dim0,conv2d_1.dim1,conv2d_1.dim2)
    # print("CONV_1:")
    # conv2d_1.__printarray__()

    # # var maxpool1 = maxpool(conv2d_1,2)
    # # print("Dimentions after 1st maxpool: ", maxpool1.dim0,maxpool1.dim1,maxpool1.dim2)
    # # print("maxpool1:")
    # # maxpool1.__printarray__()

    # var conv2d_2 = Conv2D(conv2d_1,conv2D_weights[1], 1, 1, baises[1])
    # print("Dimentions after 2nd conv: ", conv2d_2.dim0,conv2d_2.dim1,conv2d_2.dim2)
    # print("CONV_2:")
    # conv2d_2.__printarray__()

    # var maxpool2 = maxpool(conv2d_2,2)
    # print("Dimentions after 2nd maxpool: ", maxpool2.dim0,maxpool2.dim1,maxpool2.dim2)
    # print("maxpool2:")
    # maxpool2.__printarray__()

    # var conv2d_3 = Conv2D(maxpool2,conv2D_weights[2], 1, 1, baises[2])
    # print("Dimentions after 3rd conv: ", conv2d_3.dim0,conv2d_3.dim1,conv2d_3.dim2)
    # print("CONV_3:")
    # conv2d_3.__printarray__()

    # # var maxpool3 = maxpool(conv2d_3,2)
    # # print("Dimentions after 3rd maxpool: ", maxpool3.dim0,maxpool3.dim1,maxpool3.dim2)
    # # print("maxpool3:")
    # # maxpool3.__printarray__()

    # var conv2d_4 = Conv2D(conv2d_3,conv2D_weights[3], 1, 1, baises[3])
    # print("Dimentions after 4th conv: ", conv2d_4.dim0,conv2d_4.dim1,conv2d_4.dim2)
    # print("CONV_4:")
    # conv2d_4.__printarray__()

    # var maxpool4 = maxpool(conv2d_4,2)
    # print("Dimentions after 4th maxpool: ", maxpool4.dim0,maxpool4.dim1,maxpool4.dim2)
    # print("maxpool4:")
    # maxpool4.__printarray__()


    # var flat = flatten3(maxpool4)
    # print("Dimentions after flatten: ", flat.dim0, flat.dim1)
    # print("FLAT")
    # flat.__printarray__()

    # var dense_1 = dense(flat,dense_weights[0], baises[4])
    # var dense_1_act = relu(dense_1)
    # print("Dimentions after dense1: ", dense_1_act.dim0, dense_1_act.dim1)
    # print("DENSE_1")
    # dense_1_act.__printarray__()

    # var dense_2 = dense(dense_1_act,dense_weights[1], baises[5])
    # var dense_2_act = relu(dense_2)
    # print("Dimentions after dense2: ", dense_2_act.dim0, dense_2_act.dim1)
    # print("DENSE_2")
    # dense_2_act.__printarray__()

    # var dense_3 = dense(dense_2_act,dense_weights[2], baises[6])
    # var dense_3_act = sigmoid(dense_3)
    # print("Dimentions after dense3: ", dense_3_act.dim0, dense_3_act.dim1)
    # print("DENSE_3")
    # dense_3_act.__printarray__()
