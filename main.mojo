from DataStructure.Array2D import Array2D, Array3D
from DataStructure.Array1D import Array1D
from ImageProcessing.read import read_image
from ImageProcessing.store import save_image
from ImageProcessing.write import write_image
from ImageProcessing.normalization import normalize
from Layers.Conv2D import Conv2D, pading
from Layers.Maxpooling import maxpool
from Layers.Flatten import flatten, flatten2
from Layers.Dense import dense
from Activations.Relu import relu
from Activations.Sigmoid import sigmoid
# from Parser.weights import collect_weights, collect_conv2d_weights, collect_dense_weights, collect_baises
from Parser.mojo_weights import collect_weights
import math

fn main() raises:
    # # Reading Input image using opencv
    # var image_path = "Example/example.jpg"
    # var image = read_image(image_path)
    # var img = Array2D(image.shape[0].__index__() , image.shape[1].__index__())

    # # Saving image in the data structure created
    # save_image(img,image)
    # print("Input image shape: ", img.dim0, img.dim1)


#======================DUMMY EXAMPLE==========================
    # var image = Array3D(2, 6,6)
    # for i in range(image.dim0):
    #     for j in range(image.dim1):
    #         for k in range(image.dim2):
    #             if i == 1:
    #                 image.__setitem__(i,j,k,j+k+1)
    #             else:
    #                 image.__setitem__(i,j,k,j+k)
    # print("Image:")
    # image.__printarray__()
    # print("image.shape:", image.dim0, image.dim1, image.dim2)

    # var filter1 = Array3D(4,3,3)
    # filter1.__setitem__(0,0,0,1)
    # filter1.__setitem__(0,0,1,1)
    # filter1.__setitem__(0,0,2,1)
    # filter1.__setitem__(0,1,0,1)
    # filter1.__setitem__(0,1,1,-18)
    # filter1.__setitem__(0,1,2,1)
    # filter1.__setitem__(0,2,0,1)
    # filter1.__setitem__(0,2,1,1)
    # filter1.__setitem__(0,2,2,1)

    # filter1.__setitem__(1,0,0,1)
    # filter1.__setitem__(1,0,1,1)
    # filter1.__setitem__(1,0,2,1)
    # filter1.__setitem__(1,1,0,1)
    # filter1.__setitem__(1,1,1,1)
    # filter1.__setitem__(1,1,2,1)
    # filter1.__setitem__(1,2,0,1)
    # filter1.__setitem__(1,2,1,1)
    # filter1.__setitem__(1,2,2,1)

    # filter1.__setitem__(2,0,0,1)
    # filter1.__setitem__(2,0,1,1)
    # filter1.__setitem__(2,0,2,1)
    # filter1.__setitem__(2,1,0,1)
    # filter1.__setitem__(2,1,1,1)
    # filter1.__setitem__(2,1,2,1)
    # filter1.__setitem__(2,2,0,1)
    # filter1.__setitem__(2,2,1,1)
    # filter1.__setitem__(2,2,2,1)

    # filter1.__setitem__(3,0,0,1)
    # filter1.__setitem__(3,0,1,1)
    # filter1.__setitem__(3,0,2,1)
    # filter1.__setitem__(3,1,0,1)
    # filter1.__setitem__(3,1,1,-17)
    # filter1.__setitem__(3,1,2,1)
    # filter1.__setitem__(3,2,0,1)
    # filter1.__setitem__(3,2,1,1)
    # filter1.__setitem__(3,2,2,1)


    # var hmm = DynamicVector[Array3D] (capacity = 2)
    # hmm.append(filter1)
    # hmm.append(filter1)

    # print("Filter1:")
    # filter1.__printarray__()
    # print("filter1.shape:", filter1.dim0, filter1.dim1, filter1.dim2)

    # var filterbias1 = DynamicVector[Float32] (capacity = 2)
    # filterbias1.append(-0.2)
    # filterbias1.append(1.3)

    # var dense_weights = Array2D(18,1)
    # for i in range(dense_weights.dim0):
    #     for j in range(dense_weights.dim1):
    #         dense_weights.__setitem__(i,j,1.0)

    # print("dense_weights:")
    # dense_weights.__printarray__()
    # print("dense_weights.shape:", dense_weights.dim0, dense_weights.dim1)
    
    # var densebias1 = DynamicVector[Float32] (capacity = 1)
    # densebias1.append(0.7)

    # # Conv2D layer
    # var a = Conv2D(image, hmm[0], 1, 1, filterbias1)
    # print("a(1st conv2D):")
    # a.__printarray__()
    # print("a.shape:", a.dim0, a.dim1, a.dim2)

    # var b:Array3D = relu(a)
    # print("b:")
    # b.__printarray__()
    # print("b.shape:", b.dim0, b.dim1, b.dim2)

    # # Conv2D layer
    # var c = Conv2D(b, hmm[1], 1, 1, filterbias1)
    # print("c(2nd conv2D):")
    # c.__printarray__()
    # print("c.shape:", c.dim0, c.dim1, c.dim2)

    # var d:Array3D = relu(c)
    # print("d:")
    # d.__printarray__()
    # print("d.shape:", d.dim0, d.dim1, d.dim2)

    # #Maxpool
    # var e:Array3D = maxpool(d,2)
    # print("e(maxpool):")
    # e.__printarray__()
    # print("e.shape:", e.dim0, e.dim1, e.dim2)

    # #Flatten
    # var f:Array2D = flatten(e)
    # print("f(flatten):")
    # f.__printarray__()
    # print("f.shape:", f.dim0, f.dim1)

    # #Dense
    # var g:Array2D = dense(f, dense_weights, densebias1)
    # print("g(dense):")
    # g.__printarray__()
    # print("g.shape:", g.dim0, g.dim1)

    # var h:Array2D = sigmoid(g)
    # print("h(sigmoid):")
    # h.__printarray__()
    # print("h.shape:", h.dim0, h.dim1)
    
    # collect_weights()

    var image_path = "Example/car.jpg"
    var image = read_image(image_path)
    
    var img = Array3D(image.shape[2].__index__(),image.shape[0].__index__() , image.shape[1].__index__())
    save_image(img,image)
    print("Input image shape: ", img.dim0, img.dim1,img.dim2)

    var conv2D_weights = DynamicVector[Array3D] ()
    var dense_weights = DynamicVector[Array2D] ()
    var baises = DynamicVector[DynamicVector[Float32]]()
    # collect_conv2d_weights(conv2D_weights)
    # collect_dense_weights(dense_weights)
    # collect_baises(baises)

    # print("CONV2D WEIGHTS")
    # for i in range(len(conv2D_weights)):
    #     conv2D_weights[i].__printarray__()
    # print("DENSE WEIGHTS")
    # for i in range(len(dense_weights)):
    #     dense_weights[i].__printarray__()
    # print("BAISES")
    # for i in range(len(baises)):
    #     for j in range(len(baises[i])):
    #         print(baises[i][j])

    # print("IMAGE:")
    # img.__printarray__()
    # var conv2d_1 = Conv2D(img,conv2D_weights[0], 1, 1, baises[0])
    # print("Dimentions after 1st conv: ", conv2d_1.dim0,conv2d_1.dim1,conv2d_1.dim2)
    # print("CONV_1:")
    # conv2d_1.__printarray__()

    # var conv2d_2 = Conv2D(conv2d_1,conv2D_weights[1], 1, 1, baises[1])
    # print("Dimentions after 1st conv: ", conv2d_2.dim0,conv2d_2.dim1,conv2d_2.dim2)
    # print("CONV_2:")
    # conv2d_2.__printarray__()

    # var maxpool_1 = maxpool(conv2d_2,2)
    # print("Dimentions after 1st maxpool: ", maxpool_1.dim0,maxpool_1.dim1,maxpool_1.dim2)
    # print("MAXPOOL_1")
    # maxpool_1.__printarray__()

    # var flat = flatten2(maxpool_1)
    # print("Dimentions after flatten: ", flat.dim0, flat.dim1)
    # print("FLAT")
    # flat.__printarray__()
    

    # var dense_1 = dense(flat,dense_weights[0], baises[2])
    # var dense_1_act = relu(dense_1)
    # print("Dimentions after dense1: ", dense_1_act.dim0, dense_1_act.dim1)
    # print("DENSE_1")
    # dense_1_act.__printarray__()

    # var dense_2 = dense(dense_1_act,dense_weights[1], baises[3])
    # var dense_2_act = sigmoid(dense_2)
    # print("Dimentions after dense2: ", dense_2_act.dim0, dense_2_act.dim1)
    # print("DENSE_2")
    # dense_2_act.__printarray__()

    # print("SHAPES")
    # print(dense_weights[0].dim0, dense_weights[0].dim1)
    # print(dense_weights[1].dim0, dense_weights[1].dim1)
    # print(len(baises[3]))
    


    collect_weights(conv2D_weights, dense_weights, baises)

    print("IMAGE:")
    img.__printarray__()

    var conv2d_1 = Conv2D(img,conv2D_weights[0], 1, 1, baises[0])
    print("Dimentions after 1st conv: ", conv2d_1.dim0,conv2d_1.dim1,conv2d_1.dim2)
    print("CONV_1:")
    conv2d_1.__printarray__()

    var conv2d_2 = Conv2D(conv2d_1,conv2D_weights[1], 1, 2, baises[1])
    print("Dimentions after 2nd conv: ", conv2d_2.dim0,conv2d_2.dim1,conv2d_2.dim2)
    print("CONV_2:")
    conv2d_2.__printarray__()

    var conv2d_3 = Conv2D(conv2d_2,conv2D_weights[2], 1, 1, baises[2])
    print("Dimentions after 3rd conv: ", conv2d_3.dim0,conv2d_3.dim1,conv2d_3.dim2)
    print("CONV_3:")
    conv2d_3.__printarray__()

    var conv2d_4 = Conv2D(conv2d_3,conv2D_weights[3], 1, 1, baises[3])
    print("Dimentions after 4th conv: ", conv2d_4.dim0,conv2d_4.dim1,conv2d_4.dim2)
    print("CONV_4:")
    conv2d_4.__printarray__()

    var maxpool_1 = maxpool(conv2d_4,2)
    print("Dimentions after 1st maxpool: ", maxpool_1.dim0,maxpool_1.dim1,maxpool_1.dim2)
    print("MAXPOOL_1")
    maxpool_1.__printarray__()

    var flat = flatten2(maxpool_1)
    print("Dimentions after flatten: ", flat.dim0, flat.dim1)
    print("FLAT")
    flat.__printarray__()

    var dense_1 = dense(flat,dense_weights[0], baises[4])
    var dense_1_act = relu(dense_1)
    print("Dimentions after dense1: ", dense_1_act.dim0, dense_1_act.dim1)
    print("DENSE_1")
    dense_1_act.__printarray__()

    var dense_2 = dense(dense_1_act,dense_weights[1], baises[5])
    var dense_2_act = relu(dense_2)
    print("Dimentions after dense2: ", dense_2_act.dim0, dense_2_act.dim1)
    print("DENSE_2")
    dense_2_act.__printarray__()


    var dense_3 = dense(dense_2_act,dense_weights[2], baises[6])
    var dense_3_act = sigmoid(dense_3)
    print("Dimentions after dense2: ", dense_3_act.dim0, dense_3_act.dim1)
    print("DENSE_3")
    dense_3_act.__printarray__()
