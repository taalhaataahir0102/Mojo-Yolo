from DataStructure.TensorVector import Array2D
from ImageProcessing.read import read_image
from ImageProcessing.store import save_image
from ImageProcessing.write import write_image
from ImageProcessing.normalization import normalize
from Layers.Conv2D import Conv2D, pading

fn main() raises:
    # Reading Input image using opencv
    var image_path = "Example/example.jpg"
    var image = read_image(image_path)
    var img = Array2D(image.shape[0].__index__() , image.shape[1].__index__())

    # Saving image in the data structure created
    save_image(img,image)
    print("Input image shape: ", img.dim0, img.dim1)

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

    # Applying Convolution
    var output:Array2D = Conv2D(img,fltr,20,1)
    print("Image shape after conv ", output.dim0, output.dim1)

    # Normalizing Image
    var r:Array2D = normalize(output)
    write_image(r,"output.jpeg")


