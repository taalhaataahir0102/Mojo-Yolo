from DataStructure.Array2D import Array2D

fn sigmoid(img:Array2D) -> Array2D:
    var new_img:Array2D = Array2D(img.dim0,img.dim1)
    for i in range(new_img.dim0):
        for j in range(new_img.dim1):
            new_img.__setitem__(i,j, 1/(1+math.exp(-1*img.__getitem__(i,j))))
    return new_img