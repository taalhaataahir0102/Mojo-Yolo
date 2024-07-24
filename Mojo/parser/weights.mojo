from tensor import Tensor
from python import Python
from python import Python as py
from utils.index import Index
from Datastructure.Array2D import Array4D, Matrix, Array1D

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
        if (layers[i].__contains__("dense") and layers[i].__contains__("kernel")) or (layers[i].__contains__("classifier_head") and layers[i].__contains__("kernel")) or (layers[i].__contains__("regressor_head") and layers[i].__contains__("kernel")):
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

fn collect_weights(inout conv2D_weights:List[Array4D], inout dense_weights:List[Matrix], inout baises: List[Array1D]) raises:
    print("Collecting weights")
    # var input = py.import_module("builtins").input
    Python.add_to_path("/home/talha/Downloads/10xlaptop/10xlaptop/fast_mojo_yolo/parser")
    var mypython = Python.import_module("weights")
    var dict: PythonObject = mypython.read_hdf5("/home/talha/Downloads/10xlaptop/10xlaptop/fast_mojo_yolo/parser/model_weights.h5")
    var keys = dict.keys()
    var s: String = keys.__str__()
    var l:Int = keys.__len__()  
    var layers = List[String] (capacity = l)
    layers = layers_manipulation(s,l)
    var items = dict.values()
    var s2: String = items.__str__()
    var l2:Int = items.__len__()
    var shapes = List[String] (capacity = l)
    shapes = shape_manipulation(s2,l2)

    for i in range(len(shapes)):       
        var split_values = split_string(shapes[i], ",")
        var int_sizes = string_vec_to_int(split_values)
        
        if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
            var new_filter = Array4D (int_sizes[0], int_sizes[1], int_sizes[2], int_sizes[3])
            conv2D_weights.append(new_filter)
        elif layers[i].__contains__("bias"):
            var new_bias = Array1D (int_sizes[0])
            baises.append(new_bias)
        if (layers[i].__contains__("dense") and layers[i].__contains__("kernel")) or (layers[i].__contains__("classifier_head") and layers[i].__contains__("kernel")) or (layers[i].__contains__("regressor_head") and layers[i].__contains__("kernel")):
            var new_dense = Matrix (int_sizes[0], int_sizes[1])
            dense_weights.append(new_dense)  
    
    var conv2D_index:Int = 0
    var dense_index:Int = 0
    var bias_index:Int = 0
    var total_trainable_wights:Int = 0

    for i in range(len(layers)):
        var specific_weight: PythonObject = mypython.read2_hdf5("/home/talha/Downloads/10xlaptop/10xlaptop/fast_mojo_yolo/parser/model_weights.h5", layers[i])
        if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
            conv2D_weights[conv2D_index] = conv2D_weights[conv2D_index].from_numpy(specific_weight)
            total_trainable_wights += conv2D_weights[conv2D_index].total_dims()
            conv2D_index +=1

        if (layers[i].__contains__("dense") and layers[i].__contains__("kernel")) or (layers[i].__contains__("classifier_head") and layers[i].__contains__("kernel")) or (layers[i].__contains__("regressor_head") and layers[i].__contains__("kernel")):
            dense_weights[dense_index] = dense_weights[dense_index].from_numpy(specific_weight)
            total_trainable_wights += dense_weights[dense_index].total_dims()
            dense_index+=1
        
        if layers[i].__contains__("bias"):
            baises[bias_index] = baises[bias_index].from_numpy(specific_weight)
            total_trainable_wights += baises[bias_index].total_dims()
            bias_index+=1

    print("layers; shapes:")
    for i in range(len(layers)):
        print(layers[i],";", shapes[i])

    print("total_conv_layers:", len(conv2D_weights))
    print("total_dense_layers:", len(dense_weights))
    print("total_bias_layers:", len(baises))
    print("total_trainable_wights:", total_trainable_wights)


