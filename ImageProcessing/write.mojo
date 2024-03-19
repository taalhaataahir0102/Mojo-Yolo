from python import Python
from DataStructure.Array2D import Array2D

def write_image(img:Array2D, out:String) -> None:
    var cv = Python.import_module("cv2")
    var np = Python.import_module("numpy")
    var image_array = np.zeros((img.dim0, img.dim1))
    for i in range(img.dim0):
        for j in range(img.dim1):
            image_array.itemset((i, j), img.__getitem__(i, j))
    cv.imwrite(out, image_array)
