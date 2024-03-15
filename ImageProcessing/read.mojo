from python import Python

def read_image(image_path: String) -> PythonObject:
    var cv = Python.import_module("cv2")
    var image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    # image = cv.resize(image, (200,200))
    return image
