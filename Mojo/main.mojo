from tensor import Tensor
from Imageprocessing.read import read_image, read_image_mask
from Imageprocessing.save import save_image, save_image_mask
from parser.weights import collect_weights
from utils.index import Index
from python import Python
from Layers.conv import Conv2Dim2col, matmul_tiled, Conv2D
from Layers.maxpool import maxpool
from Layers.dense import dense
from Datastructure.Array2D import Array4D, Matrix, Array1D
from time import now
import pathlib
import os
from max.graph import Graph, TensorType, Type, ops
from max import engine


fn yolo_model(input_img: Array4D, inout conv2D_weights:List[Array4D], dense_weights:List[Matrix],baises:List[Array1D], inout time_in_matmul: Float32) raises -> List[Matrix]:
    var layer1  = Conv2Dim2col(input_img, conv2D_weights[0], baises[1], time_in_matmul, 1, 1)
    print("layer1.shape: ")
    layer1.__shape__()

    var layer2  = Conv2Dim2col(layer1, conv2D_weights[1], baises[2], time_in_matmul,1, 0)
    print("layer2.shape: ")
    layer2.__shape__()

    var layer3  = Conv2Dim2col(layer2, conv2D_weights[2], baises[3],time_in_matmul, 1, 0)
    print("layer3.shape: ")
    layer3.__shape__()

    var layer4  = maxpool(layer3,2, 2)
    print("layer4.shape: ")
    layer4.__shape__()

    var layer5  = Conv2Dim2col(layer4, conv2D_weights[3], baises[4],time_in_matmul, 1, 0)
    print("layer5.shape: ")
    layer5.__shape__()

    var layer6  = Conv2Dim2col(layer5, conv2D_weights[4], baises[5],time_in_matmul, 1, 0)
    print("layer6.shape: ")
    layer6.__shape__()

    var layer7  = Conv2Dim2col(layer6, conv2D_weights[5], baises[6],time_in_matmul, 2, 0)
    print("layer7.shape: ")
    layer7.__shape__()

    var layer8  = maxpool(layer7,2, 2)
    print("layer8.shape: ")
    layer8.__shape__()

    var layer9  = Conv2Dim2col(layer8, conv2D_weights[6], baises[7],time_in_matmul, 1, 0)
    print("layer9.shape: ")
    layer9.__shape__()

    var layer10  = Conv2Dim2col(layer9, conv2D_weights[7], baises[8],time_in_matmul, 1, 0)
    print("layer10.shape: ")
    layer10.__shape__()

    var layer11  = Conv2Dim2col(layer10, conv2D_weights[8], baises[9],time_in_matmul, 1, 0)
    print("layer11.shape: ")
    layer11.__shape__()

    var layer12  = Conv2Dim2col(layer11, conv2D_weights[9], baises[10],time_in_matmul, 1, 0)
    print("layer12.shape: ")
    layer12.__shape__()

    var layer13  = Conv2Dim2col(layer12, conv2D_weights[10], baises[11],time_in_matmul, 1, 0)
    print("layer13.shape: ")
    layer13.__shape__()

    var layer14:Matrix = layer13.__reshape2__(1,layer13.dim0*layer13.dim1*layer13.dim2*layer13.dim3)
    print("layer14.shape: ")
    layer14.__shape__()
    
    var layer15 = Matrix(layer14.rows, dense_weights[1].cols)
    var start_time = now()
    matmul_tiled(layer15, layer14, dense_weights[1],baises[12])
    var end_time = now()
    var elapsed_time = (end_time - start_time)/ 1000000000
    time_in_matmul += elapsed_time
    print("layer15.shape: ")
    layer15.__shape__()

    var layer16 = Matrix(layer15.rows, dense_weights[2].cols)
    start_time = now()
    matmul_tiled(layer16, layer15, dense_weights[2],baises[13])
    end_time = now()
    elapsed_time = (end_time - start_time)/ 1000000000
    time_in_matmul += elapsed_time
    print("layer16.shape: ")
    layer16.__shape__()

    var layer17 = Matrix(layer16.rows, dense_weights[3].cols)
    start_time = now()
    matmul_tiled(layer17, layer16, dense_weights[3],baises[14])
    end_time = now()
    elapsed_time = (end_time - start_time)/ 1000000000
    time_in_matmul += elapsed_time
    print("layer17.shape: ")
    layer17.__shape__()

    var layer18 = Matrix(layer16.rows, dense_weights[0].cols)
    start_time = now()
    matmul_tiled(layer18, layer17, dense_weights[0], baises[0], "sigmoid")
    end_time = now()
    elapsed_time = (end_time - start_time)/ 1000000000
    time_in_matmul += elapsed_time
    print("layer18.shape: ")
    layer18.__shape__()

    var output = List[Matrix] ()
    output.append(layer17)
    output.append(layer18)
    return output

fn yolo_model(input_img: Array4D, inout conv2D_weights:List[Array4D], dense_weights:List[Matrix],
              baises:List[Array1D]) raises -> Matrix:
    ###### Convoluton #####
    var layer1  = Conv2D(input_img, conv2D_weights[0], baises[0], stride =1, padding = "same", activation = "relu")
    var layer2  = Conv2D(layer1, conv2D_weights[1], baises[1],stride = 1, padding = "valid", activation = "relu")
    var layer3  = Conv2D(layer2, conv2D_weights[2], baises[2],stride = 2, padding = "same", activation = "relu")
    var layer4  = maxpool(layer3, 2, 2)
    ##### Flatten #####
    var layer5:Matrix = layer4.__reshape2__(1,layer4.dim0*layer4.dim1*layer4.dim2*layer4.dim3)
    ##### Dense #####
    var layer6 = Matrix(layer5.rows, dense_weights[0].cols)
    matmul_tiled(layer6, layer5, dense_weights[0],baises[3], act ="lrelu")
    var layer7 = Matrix(layer6.rows, dense_weights[1].cols)
    matmul_tiled(layer7, layer6, dense_weights[1],baises[4], act = "sigmoid")
    return layer7


def main():
    var time_in_matmul:Float32 = 0.0
    var conv2D_weights = List[Array4D] ()
    var dense_weights = List[Matrix] ()
    var baises = List[Array1D] ()

    collect_weights(conv2D_weights, dense_weights, baises)
    print("Weights Collected")


    var directory = pathlib.Path("Example/input")
    var image_names = List[String] ()
    var filenames = directory.listdir()
    for i in range(len(filenames)):
        var filename = filenames[i]
        var file_path = directory / filename
        if file_path.is_file():
            image_names.append(str(filename))
    
    print("#####RUN MODEL#####")

    var image_path:String = "Example/mask.jpeg"
    # var image_path = string1 + image_names[0]
    print(image_path)
    var image = read_image_mask(image_path)
    var ten = Array4D(image.shape[0].__int__(),image.shape[1].__int__(),image.shape[2].__int__(),1)
    var tens = ten.from_numpy(image)
    
    tens.__shape__()

    var result = yolo_model(tens, conv2D_weights, dense_weights, baises, time_in_matmul)

    var predicted_label:PythonObject = result[1].to_numpy()[0]
    var predicted_box:PythonObject = result[0].to_numpy()[0]

    print(predicted_label, predicted_box)


    Python.add_to_path("/home/talha/Downloads/10xlaptop/10xlaptop/fast_mojo_yolo")
    var mypython = Python.import_module("draw")
    mypython.process_and_predict(image_path, 144, predicted_label, predicted_box, image_names[0])

    print("#####LIVE#####")


    var width = 100
    var height = 100
    var video_stream_url = "http://192.168.18.5:8080/video"
    var cv = Python.import_module("cv2")
    var cap = cv.VideoCapture(video_stream_url)
    if not cap.isOpened():
        print("Error: Unable to open video stream.")
    var frame_count = 0

    # Create a named window
    cv.namedWindow('Live Video', cv.WINDOW_NORMAL)
    # Resize the window
    cv.resizeWindow('Live Video', 500, 400)

    var start_time = now()
    time_in_matmul = 0.0
    while True:
        var starti = now()
        var frame = cap.read()
        if not frame[0] or ((now() - start_time)/1000000000) > 20:
            print("Error: Unable to read frame.")
            break
        var frame_filename = "video/frame_" + str(frame_count) + ".jpg"

        var resized_frame = cv.resize(frame[1], (width, height))
        cv.imwrite(frame_filename, resized_frame)

        var end34 = now()
        var execution_time_frame_reading = (end34 - starti)
        var execution_time_seconds_frame_reading :  Float32 = execution_time_frame_reading / 1000000000
        print("Frame Reading:", execution_time_seconds_frame_reading)
        
        var start = now()

        var image = read_image_mask(frame_filename)
        var ten = Array4D(image.shape[0].__int__(),image.shape[1].__int__(),image.shape[2].__int__(),1)
        var tens = ten.from_numpy(image)

        var end = now()
        var execution_time_preprocessing = (end - start)
        var execution_time_seconds_preprocessing :  Float32 = execution_time_preprocessing / 1000000000
        print("Image Preprocessing:", execution_time_seconds_preprocessing)

        var start1 = now()

        var result = yolo_model(tens, conv2D_weights, dense_weights, baises, time_in_matmul)
        var predicted_label:PythonObject = result[1].to_numpy()[0]
        var predicted_box:PythonObject = result[0].to_numpy()[0]

        var end1 = now()
        var execution_time_modelexecution = (end1 - start1)
        var execution_time_seconds_modelexecution :  Float32 = execution_time_modelexecution / 1000000000
        print("Model Execution:", execution_time_seconds_modelexecution)
        
        var start2 = now()

        Python.add_to_path("/home/talha/Downloads/10xlaptop/10xlaptop/fast_mojo_yolo")
        var mypython = Python.import_module("draw")
        var show: PythonObject = mypython.process_and_predict(frame_filename, 100, predicted_label, predicted_box, str(frame_count) + ".jpg")

        var end2 = now()
        var execution_time_postprocessing = (end2 - start2)
        var execution_time_seconds_postprocessing :  Float32 = execution_time_postprocessing / 1000000000
        print("Post Processing:", execution_time_seconds_postprocessing)
        frame_count += 1

        cv.imshow('Live Video', show)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        var endi = now()
        var execution_time = (endi - starti)
        var execution_time_seconds :  Float32 = execution_time / 1000000000
        print("total time:", execution_time_seconds)
        print("======================================================")

    print("time_in_matmul:", time_in_matmul)
    var end_time = now()
    var elapsed_time = (end_time - start_time)/ 1000000000
    print("total time:", elapsed_time)
    var fps = frame_count / elapsed_time

    print("Total frames:", frame_count)
    print("Frame rate:", fps)


    cap.release()
    cv.destroyAllWindows()