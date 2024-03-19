from DataStructure.Array2D import Array2D
from DataStructure.Array1D import Array1D
from ImageProcessing.read import read_image
from ImageProcessing.store import save_image
from ImageProcessing.write import write_image
from ImageProcessing.normalization import normalize
from Layers.Conv2D import Conv2D, pading
from Layers.Maxpooling import maxpool
from Layers.Flatten import flatten
from Layers.Dense import dense

fn main() raises:
    # Reading Input image using opencv
    var image_path = "Example/example.jpg"
    var image = read_image(image_path)
    var img = Array2D(image.shape[0].__index__() , image.shape[1].__index__())

    # Saving image in the data structure created
    save_image(img,image)
    print("Input image shape: ", img.dim0, img.dim1)

    #Checking Max pool layer
    # img.__printarray__()
    # print("img shape before max pooling: ", img.dim0, img.dim1)
    # var n:Array2D = maxpool(img,2)
    # print("img shape after max pooling: ", n.dim0, n.dim1)
    # n.__printarray__()

    # Creating filter
    var fltr = Array2D(3,3)
    fltr.__setitem__(0,0,1)
    fltr.__setitem__(0,1,0)
    fltr.__setitem__(0,2,-1)
    fltr.__setitem__(1,0,1)
    fltr.__setitem__(1,1,0)
    fltr.__setitem__(1,2,-1)
    fltr.__setitem__(2,0,1)
    fltr.__setitem__(2,1,0)
    fltr.__setitem__(2,2,-1)
    print("filter")
    fltr.__printarray__()

    # Checking flatten layer
    # var f:Array2D = flatten(fltr)
    # f.__printarray__()
    
    # Applying Convolution
    var output:Array2D = Conv2D(img,fltr,1,1,0.0)
    print("Image shape after conv ", output.dim0, output.dim1)

    # Normalizing Image
    var r:Array2D = normalize(output)
    write_image(r,"Example/output.jpeg")

    # Checking Dense layer
    # var a = Array2D(2,3)
    # var b = Array2D(3,1)
    # for i in range(a.dim0):
    #     for j in range(a.dim1):
    #         if i == 0:
    #             a.__setitem__(i,j,2)
    #         else:
    #             a.__setitem__(i,j,3)
    # for i in range(b.dim0):
    #     for j in range(b.dim1):
    #         b.__setitem__(i,j,5)
    #         if i == 2:
    #             b.__setitem__(i,j,6)
    # print("a:")
    # a.__printarray__()
    # print("b:")
    # b.__printarray__()

    # var o:Array2D = dense(a,b,0.0)

    # print("o after mult:")
    # o.__printarray__()


