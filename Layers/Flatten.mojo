from DataStructure.Array2D import Array2D

fn flatten(img: Array2D) -> Array2D:
    var flat = Array2D(1,img.dim0*img.dim1)
    for i in range(img.dim0):
        for j in range(img.dim1):
            flat.__setitem__(0,(i*img.dim1)+j, img.__getitem__(i,j))
    return flat
