import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def process_and_predict(image_path, input_size, predicted_label, predicted_box,img_name):
    # Load the image
    original_image = cv.imread(image_path)
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Preprocess the image (resize, normalize, etc.) for model input
    resized_image = cv.resize(gray_image, (input_size, input_size))
    normalized_image = resized_image.astype(np.float32) / 255.0  # Normalize the pixel values

    # Expand dimensions to match the shape expected by the model
    image_input = np.expand_dims(normalized_image, axis=0)
    image_input = np.expand_dims(image_input, axis=-1)

    # Convert predicted box to correct scale
    predicted_box = predicted_box * np.array([original_image.shape[1], original_image.shape[0], original_image.shape[1], original_image.shape[0]])
    predicted_box = tf.cast(predicted_box, tf.int32).numpy()  # Convert to numpy array
    x1, y1, x2, y2 = predicted_box

    # Draw predicted bounding box on the original image
    if predicted_label[0] > predicted_label[1]:
        color = (0, 255, 0)  # Green color
    else:
        color = (0, 0, 255)  # Red color

    cv.rectangle(original_image, (x1, y1), (x2, y2), color, 2)

    # Save the image with the predicted bounding box
    output_image_path = 'output_stream/' + img_name
    cv.imwrite(output_image_path, original_image)

    # print("original_image:", original_image.shape)

    return original_image



# Example usage
# image_path = 'Example/mask.jpeg'
# input_size = 100
# predicted_label = np.array([0.88094532, 0.54024714])
# predicted_box = np.array([0.30740494, 0.16525207, 0.56029695, 0.51501477])
# process_and_predict(image_path, input_size, predicted_label, predicted_box, "predicted.jpg")
