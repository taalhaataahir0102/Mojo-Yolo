from max.engine import InputSpec, InferenceSession
from python import Python
from tensor import TensorSpec, Tensor
from utils.index import Index
from time import now


@always_inline
fn numpy_data_pointer[
    type: DType
](numpy_array: PythonObject) raises -> DTypePointer[type]:
    return DTypePointer[type](
        address=int(numpy_array.__array_interface__["data"][0])
    )

@always_inline
fn tensor_to_numpy[
    type: DType
](tensor: Tensor[type]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var tensor_shape = tensor.shape()
    var tensor_rank = tensor.rank()

    var python_list = Python.evaluate("list()")
    for i in range(tensor_rank):
        _ = python_list.append(tensor_shape[i])

    var numpy_array:PythonObject = np.zeros(python_list, dtype=np.float32)
    var dst = numpy_data_pointer[type](numpy_array)
    var src = tensor.unsafe_ptr()
    var length = tensor.num_elements()
    memcpy(dst, src, length)

    return numpy_array

@always_inline
fn numpy_to_tensor(numpy_array: PythonObject) raises -> Tensor[DType.float32]:
    
    var tensor_shape = numpy_array.shape
    var tensor_rank = len(numpy_array.shape)

    var shape_list: List[Int]  = List[Int]()
    for i in range(tensor_rank):
        shape_list.append(tensor_shape[i].__int__())

    var tensor = Tensor[DType.float32] (shape_list)

    var src = numpy_data_pointer[DType.float32](numpy_array)
    var dst = tensor.unsafe_ptr()
    var length = tensor.num_elements()
    memcpy(dst, src, length)
    return tensor

fn pre_processing(s:String) raises -> Tensor[DType.float32]:
    var transformers = Python.import_module("transformers")
    var image_processor = transformers.YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    Python.add_to_path(".")
    var mypython = Python.import_module("helper")
    var image: PythonObject = mypython.pre_processing(s, image_processor)
    var tens = numpy_to_tensor(image)
    return tens


fn main() raises:
    var transformers = Python.import_module("transformers")
    var image_processor = transformers.YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    var model1 = transformers.YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    var model_path = "yolos_tiny.torchscript"
    
    
    # Assuming the input image is of shape (1, 3, 224, 224)
    var input_spec = TensorSpec(DType.float32, 1, 3, 144, 144)
    var input_specs = List[InputSpec]()
    input_specs.append(input_spec)

    Python.add_to_path(".")
    var mypython = Python.import_module("helper")
    var image: PythonObject = mypython.pre_processing("test.jpg", image_processor)
    
    var tens = numpy_to_tensor(image)

    # var tens = pre_processing("test.jpg")
    var session = InferenceSession()
    var model = session.load(model_path, input_specs=input_specs)
    var outputs = model.execute("pixel_values", tens)
    var result0 = outputs.get[DType.float32]("result0")
    var result1 = outputs.get[DType.float32]("result1")
    var result2 = outputs.get[DType.float32]("result2")

    var logits = tensor_to_numpy(result0)
    var pred_boxes = tensor_to_numpy(result1)
    var last_hidden_state = tensor_to_numpy(result2)

    var post: PythonObject = mypython.post_processing(logits, pred_boxes, last_hidden_state, "test.jpg", image_processor, model1)

    print("#####live#####")
    var width = 144
    var height = 144
    var video_stream_url = "http://192.168.18.5:8080/video"
    var cv = Python.import_module("cv2")
    # var cap = cv.VideoCapture(video_stream_url)
    var cap = cv.VideoCapture(video_stream_url)
    if not cap.isOpened():
        print("Error: Unable to open video stream.")
    var frame_count = 0

    # # Create a named window
    # cv.namedWindow('Live Video', cv.WINDOW_NORMAL)
    # # Resize the window
    # cv.resizeWindow('Live Video', 500, 400)
    var total_model_execution_time:Float32 = 0
    var start_time = now()
    while True:
        var starti = now()
        var frame = cap.read()
        if not frame[0] or ((now()- start_time)/1000000000) > 10:
            print("Error: Unable to read frame.")
            break
        var frame_filename = "video/frame_" + str(frame_count) + ".jpg"
        var resized_frame = cv.resize(frame[1], (width, height))
        cv.imwrite(frame_filename, resized_frame)

        # var start = now()
        var image: PythonObject = mypython.pre_processing(frame_filename, image_processor)
        # var end = now()
        # var execution_time_preprocessing = (end - start)
        # var execution_time_seconds_preprocessing :  Float32 = execution_time_preprocessing / 1000000000
        # print("Image Preprocessing:", execution_time_seconds_preprocessing)
        var tens = numpy_to_tensor(image)

        var start1 = now()
        var outputs = model.execute("pixel_values", tens)
        var end1 = now()
        var execution_time_modelexecution = (end1 - start1)
        var execution_time_seconds_modelexecution :  Float32 = execution_time_modelexecution / 1000000000

        total_model_execution_time += execution_time_seconds_modelexecution
        # print("Model Execution:", execution_time_seconds_modelexecution)
        var result0 = outputs.get[DType.float32]("result0")
        var result1 = outputs.get[DType.float32]("result1")
        var result2 = outputs.get[DType.float32]("result2")

        var logits = tensor_to_numpy(result0)
        var pred_boxes = tensor_to_numpy(result1)
        var last_hidden_state = tensor_to_numpy(result2)

        # var start2 = now()
        var show: PythonObject = mypython.post_processing(logits, pred_boxes, last_hidden_state, frame_filename, image_processor, model1)
        # var end2 = now()
        # var execution_time_postprocessing = (end2 - start2)
        # var execution_time_seconds_postprocessing :  Float32 = execution_time_postprocessing / 1000000000
        # print("Post Processing:", execution_time_seconds_postprocessing)

        cv.imshow('YOLO Object Detection', cv.cvtColor(show, cv.COLOR_RGB2BGR))
        # Break the loop if 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

        var endi = now()
        var execution_time = (endi - starti)
        var execution_time_seconds :  Float32 = execution_time / 1000000000
        print("total time:", execution_time_seconds)
        print("======================================================")


        # cv.imshow('Live Video', cv.cvtColor(show, cv.COLOR_RGB2BGR))
        # if cv.waitKey(frame_interval) & 0xFF == ord('q'):
        #     break
    var end_time = now()
    var elapsed_time = (end_time - start_time) / 1000000000
    var fps = frame_count / elapsed_time
    print("Total Frames:", frame_count)
    print("Frame Rate:", fps)
    print("Total model execution time:", total_model_execution_time)
    print("Avg model execution time per frame:", total_model_execution_time/ frame_count)

    cap.release()
    cv.destroyAllWindows()
    

