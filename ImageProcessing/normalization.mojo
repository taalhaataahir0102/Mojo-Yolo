from DataStructure.TensorVector import Array2D

fn normalize(inout img: Array2D) -> Array2D:
    var minimum = img.__getitem__(0,0)
    var maximum = img.__getitem__(0,0)
    var result = Array2D(img.dim0,img.dim1)
    for i in range(img.dim0):
        for j in range(img.dim1):
            if img.__getitem__(i,j) > maximum:
                maximum = img.__getitem__(i,j)
            if img.__getitem__(i,j) < minimum:
                minimum = img.__getitem__(i,j)

    for i in range(img.dim0):
        for j in range(img.dim1):
            result.__setitem__(i,j,int(((img.__getitem__(i,j) - minimum) / (maximum - minimum)) * 255))
    
    return result
