from DataStructure.Array2D import Array2D, Array3D
from DataStructure.Array1D import Array1D
from ImageProcessing.read import read_image
from ImageProcessing.store import save_image
from ImageProcessing.write import write_image
from ImageProcessing.normalization import normalize
from Layers.Conv2D import Conv2D, pading
from Layers.Maxpooling import maxpool
from Layers.Flatten import flatten
from Layers.Dense import dense
from Activations.Relu import relu
from Activations.Sigmoid import sigmoid
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
    var image = Array3D(2, 6,6)
    for i in range(image.dim0):
        for j in range(image.dim1):
            for k in range(image.dim2):
                if i == 1:
                    image.__setitem__(i,j,k,j+k+1)
                else:
                    image.__setitem__(i,j,k,j+k)
    print("Image:")
    image.__printarray__()
    print("image.shape:", image.dim0, image.dim1, image.dim2)

    var filter1 = Array3D(4,3,3)
    filter1.__setitem__(0,0,0,1)
    filter1.__setitem__(0,0,1,1)
    filter1.__setitem__(0,0,2,1)
    filter1.__setitem__(0,1,0,1)
    filter1.__setitem__(0,1,1,1)
    filter1.__setitem__(0,1,2,1)
    filter1.__setitem__(0,2,0,1)
    filter1.__setitem__(0,2,1,1)
    filter1.__setitem__(0,2,2,1)

    filter1.__setitem__(1,0,0,1)
    filter1.__setitem__(1,0,1,1)
    filter1.__setitem__(1,0,2,1)
    filter1.__setitem__(1,1,0,1)
    filter1.__setitem__(1,1,1,1)
    filter1.__setitem__(1,1,2,1)
    filter1.__setitem__(1,2,0,1)
    filter1.__setitem__(1,2,1,1)
    filter1.__setitem__(1,2,2,1)

    filter1.__setitem__(2,0,0,1)
    filter1.__setitem__(2,0,1,1)
    filter1.__setitem__(2,0,2,1)
    filter1.__setitem__(2,1,0,1)
    filter1.__setitem__(2,1,1,1)
    filter1.__setitem__(2,1,2,1)
    filter1.__setitem__(2,2,0,1)
    filter1.__setitem__(2,2,1,1)
    filter1.__setitem__(2,2,2,1)

    filter1.__setitem__(3,0,0,1)
    filter1.__setitem__(3,0,1,1)
    filter1.__setitem__(3,0,2,1)
    filter1.__setitem__(3,1,0,1)
    filter1.__setitem__(3,1,1,1)
    filter1.__setitem__(3,1,2,1)
    filter1.__setitem__(3,2,0,1)
    filter1.__setitem__(3,2,1,1)
    filter1.__setitem__(3,2,2,1)


    print("Filter1:")
    filter1.__printarray__()
    print("filter1.shape:", filter1.dim0, filter1.dim1, filter1.dim2)

    var filterbias1 = DynamicVector[Float32] (capacity = 2)
    filterbias1.append(1.0)
    filterbias1.append(0.0)

    var dense_weights = Array2D(18,1)
    for i in range(dense_weights.dim0):
        for j in range(dense_weights.dim1):
            dense_weights.__setitem__(i,j,0.0)

    print("dense_weights:")
    dense_weights.__printarray__()
    print("dense_weights.shape:", dense_weights.dim0, dense_weights.dim1)
    
    var densebias1 = DynamicVector[Float32] (capacity = 1)
    densebias1.append(0.7)

    # Conv2D layer
    var a = Conv2D(image, filter1, 1, 1, filterbias1)
    print("a(1st conv2D):")
    a.__printarray__()
    print("a.shape:", a.dim0, a.dim1, a.dim2)

    var b:Array3D = relu(a)
    print("b:")
    b.__printarray__()
    print("b.shape:", b.dim0, b.dim1, b.dim2)

    # Conv2D layer
    var c = Conv2D(b, filter1, 1, 1, filterbias1)
    print("c(2nd conv2D):")
    c.__printarray__()
    print("c.shape:", c.dim0, c.dim1, c.dim2)

    var d:Array3D = relu(c)
    print("d:")
    d.__printarray__()
    print("d.shape:", d.dim0, d.dim1, d.dim2)

    #Maxpool
    var e:Array3D = maxpool(d,2)
    print("e(maxpool):")
    e.__printarray__()
    print("e.shape:", e.dim0, e.dim1, e.dim2)

    #Flatten
    var f:Array2D = flatten(e)
    print("f(flatten):")
    f.__printarray__()
    print("f.shape:", f.dim0, f.dim1)

    #Dense
    var g:Array2D = dense(f, dense_weights, densebias1)
    print("g(dense):")
    g.__printarray__()
    print("g.shape:", g.dim0, g.dim1)

    var h:Array2D = sigmoid(g)
    print("h(sigmoid):")
    h.__printarray__()
    print("h.shape:", h.dim0, h.dim1)
    
