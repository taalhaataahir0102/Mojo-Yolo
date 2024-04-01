from DataStructure.Array2D import Array2D


fn dense(img: Array2D, weights: Array2D, bias: DynamicVector[Float32]) -> Array2D:
    var output = Array2D(img.dim0,weights.dim1)
    print(output.dim0, output.dim1)
    for m in range(output.dim0):
        for n in range(output.dim1):
            var sum:Float32 = 0
            for k in range(img.dim1):
                sum += img.__getitem__(m,k) * weights.__getitem__(k,n)
            output.__setitem__(m,n,sum)
    var index:Int = 0
    for i in range(output.dim0):
        for j in range(output.dim1):
            output.__setitem__(i,j,output.__getitem__(i,j)+ bias[index])
            index+=1
    return output