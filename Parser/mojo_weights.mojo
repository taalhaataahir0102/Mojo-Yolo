from python import Python
from python import Python as py
from DataStructure.Array2D import Array2D, Array3D

fn layers_manipulation(s:String, l:Int) -> List[String]:
    var k = List[String] (capacity = l)
    var str_len:Int = s.__len__()
    var idx:Int = 0
    var found:Bool = False
    var start:String = ''
    while(idx < str_len):
        if s[idx] == "'" and found == False:
            found = True
            idx+=1
        elif found == True and s[idx] != "'":
            start += s[idx]
            idx+=1
        elif s[idx] == "'" and found == True:
            idx +=1
            k.append(start)
            found = False
            start = ''
        else:
            idx+=1
        
    return k


fn get_total_conv_layers(layers: List[String])->Int:
    var ans:Int = 0
    for i in range(len(layers)):
        if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
            ans+=1
    return ans

fn get_total_dense_layers(layers: List[String])->Int:
    var ans:Int = 0
    for i in range(len(layers)):
        if layers[i].__contains__("dense") and layers[i].__contains__("kernel"):
            ans+=1
    return ans

fn get_total_bias_layers(layers: List[String])->Int:
    var ans:Int = 0
    for i in range(len(layers)):
        if layers[i].__contains__("bias"):
            ans+=1
    return ans

fn shape_manipulation(s:String, l:Int) -> List[String]:
    var k = List[String] (capacity = l)
    var str_len:Int = s.__len__()
    var idx:Int = 0
    var found:Bool = False
    var start:String = ''
    var first:Bool = False
    while(idx < str_len):
        if s[idx] == "(" and found == False and first == True:
            found = True
            idx+=1
        elif s[idx] == "(" and found == False and first == False:
            first = True
            idx+=1
            continue
        elif found == True and s[idx] != ")":
            start += s[idx]
            idx+=1
        elif s[idx] == ")" and found == True:
            idx +=1
            k.append(start)
            found = False
            start = ''
        else:
            idx+=1
        
    return k

fn split_string(s:String, split:String) -> List[String]:
    var start:String = ''
    var k = List[String] ()
    for i in range(s.__len__()):
        if s[i] == split:
            k.append(start)
            start = ''
        else:
            start += s[i]
    if start == '':
        return k
    else:
        k.append(start)
        return k

fn string_vec_to_int (inout s:List[String]) raises -> List[Int]:
    var new_vec = List[Int] ()
    for i in range(len(s)):
        s[i] = s[i].strip()
        var n:Int = atol(s[i])
        new_vec.append(n)
    return new_vec


fn extract_number(s:String) raises -> Float32:
    Python.add_to_path("/home/lpt-10x/Desktop/Mojo-Yolo/Parser/")
    var mypython = Python.import_module("python_weights")
    var a: PythonObject = mypython.test(s)
    var b:Float32 = a.to_float64().cast[DType.float32]()
    return b

fn collect_weights(inout conv2D_weights:List[Array3D], inout dense_weights:List[Array2D], inout baises: List[List[Float32]]) raises:
    var input = py.import_module("builtins").input
    Python.add_to_path("/home/lpt-10x/Desktop/Mojo-Yolo/Parser/")
    var mypython = Python.import_module("python_weights")
    var my_dict: PythonObject = mypython.main()

    var dict: PythonObject = mypython.read_hdf5("/home/lpt-10x/Desktop/Mojo-Yolo/Parser/model_weights.h5")

    var keys = dict.keys()
    var s: String = keys.__str__()
    var l:Int = keys.__len__()
    
    var layers = List[String] (capacity = l)
    layers = layers_manipulation(s,l)

    var total_conv_layers = get_total_conv_layers(layers)
    var total_dense_layers = get_total_dense_layers(layers)
    var total_bias_layers = get_total_bias_layers(layers)


    conv2D_weights = List[Array3D] (capacity = total_conv_layers)
    dense_weights = List[Array2D] (capacity = total_dense_layers)
    baises = List[List[Float32]] (capacity = total_bias_layers)


    var items = dict.values()

    var s2: String = items.__str__()
    var l2:Int = items.__len__()

    var shapes = List[String] (capacity = l)
    shapes = shape_manipulation(s2,l2)

    for i in range(len(shapes)):
        
        var split_values = split_string(shapes[i], ",")
        var int_sizes = string_vec_to_int(split_values)
        
        if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
            var new_filter = Array3D(int_sizes[0]*int_sizes[1], int_sizes[2], int_sizes[3])
            conv2D_weights.append(new_filter)
        elif layers[i].__contains__("bias"):
            var new_bias = List[Float32] (capacity = int_sizes[0])
            baises.append(new_bias)
        elif layers[i].__contains__("dense") and layers[i].__contains__("kernel"):
            var new_dense = Array2D(int_sizes[0], int_sizes[1])
            dense_weights.append(new_dense)

    print("LAYERS:")
    for i in range(len(layers)):
        print(layers[i])

    print("CONV2D LAYERS", len(conv2D_weights))
    for i in range(len(conv2D_weights)):
        print(conv2D_weights[i].dim0,conv2D_weights[i].dim1,conv2D_weights[i].dim2)
    
    print("DENSE LAYERS", len(dense_weights))
    for i in range(len(dense_weights)):
        print(dense_weights[i].dim0,dense_weights[i].dim1)

    print("BIAS LAYERS", len(baises))
    for i in range(len(baises)):
        print(baises[i].capacity)
    

    var f = open("/home/lpt-10x/Desktop/Mojo-Yolo/Parser/weights.txt", "r")  
    var content:String = f.read()  
    f.close()

    var split_content = content.split("\n")


    var conv_index:Int = 0
    var dense_index:Int = 0
    var bias_index:Int = 0
    var conv2d_layer:Bool = False
    var bais_layer:Bool = False
    var dense_layer:Bool = False

    var loop_size:Int = 0

    print("total weights + layers",len(split_content))
    for i in range(len(split_content)):
        conv2d_layer = False
        dense_layer = False
        bais_layer = False
        if split_content[i].__contains__("conv2d") and split_content[i].__contains__("kernel"):
            loop_size = conv2D_weights[conv_index].dim0 * conv2D_weights[conv_index].dim1 * conv2D_weights[conv_index].dim2
            conv_index +=1
            conv2d_layer = True
            dense_layer = False
            bais_layer = False
            i+=1
            
        elif split_content[i].__contains__("bias"):
            loop_size = baises[bias_index].capacity
            bias_index +=1
            conv2d_layer = False
            dense_layer = False
            bais_layer = True
            i+=1

            
        elif split_content[i].__contains__("dense") and split_content[i].__contains__("kernel"):
            loop_size = dense_weights[dense_index].dim0 * dense_weights[dense_index].dim1
            dense_index +=1
            conv2d_layer = False
            dense_layer = True
            bais_layer = False
            i+=1

        for j in range(loop_size):
            if conv2d_layer == True and dense_layer == False and bais_layer == False:
                var num:Float32 = extract_number(split_content[i])
                var index0:Int = j // (conv2D_weights[conv_index-1].dim1 * conv2D_weights[conv_index-1].dim2)  # Compute the index along the first dimension
                var index1:Int = (j % (conv2D_weights[conv_index-1].dim1 * conv2D_weights[conv_index-1].dim2)) // conv2D_weights[conv_index-1].dim2  # Compute the index along the second dimension
                var index2:Int = (j % (conv2D_weights[conv_index-1].dim1 * conv2D_weights[conv_index-1].dim2)) % conv2D_weights[conv_index-1].dim2
                conv2D_weights[conv_index-1].__setitem__(index0,index1,index2,num)
            
            elif conv2d_layer == False and dense_layer == False and bais_layer == True:
                var num:Float32 = extract_number(split_content[i])
                baises[bias_index-1].append(num)

            elif conv2d_layer == False and dense_layer == True and bais_layer == False:
                var num:Float32 = extract_number(split_content[i])
                var index0:Int = j // dense_weights[dense_index-1].dim1  # Compute the index along the first dimension
                var index1:Int = j % dense_weights[dense_index-1].dim1
                dense_weights[dense_index-1].__setitem__(index0,index1,num)
            i+=1

            if i == 10000:
                print("yes")
            if i == 50000:
                print("yes2")
            if i == 100000:
                print("yes3")

        loop_size = 0

            
