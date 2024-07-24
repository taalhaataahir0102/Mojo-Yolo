from transformers import YolosImageProcessor
from PIL import Image
import torch
import cv2
import numpy as np
import time
import os

class CustomYolosObjectDetectionOutput:
    def __init__(self, logits, pred_boxes, last_hidden_state):
        self.logits = logits
        self.pred_boxes = pred_boxes
        self.last_hidden_state = last_hidden_state
        self.loss = None
        self.loss_dict = None
        self.auxiliary_outputs = None
        self.hidden_states = None
        self.attentions = None

def resize_image(image, size=(144, 144)):
    return image.resize(size, Image.BILINEAR)

def draw_boxes(image, results, labels_map, threshold=0.8):
    image = np.array(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score >= threshold:
            box = [round(i, 2) for i in box.tolist()]
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            color = (0, 0, 255)  # Red color in BGR
            thickness = 2

            # Draw rectangle
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

            # Prepare label text
            label_text = f"{labels_map[label.item()]}: {round(score.item(), 3)}"

            # Set the font scale and thickness
            font_scale = 0.5
            font_thickness = 1

            # Get the text size
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Draw the background rectangle for the text
            image = cv2.rectangle(image, start_point, (start_point[0] + text_width, start_point[1] - text_height - 4), color, -1)

            # Put the label text
            image = cv2.putText(image, label_text, (start_point[0], start_point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return image

def image_preprocessing(img_path, image_processor):
    image = Image.open(img_path)
    image = resize_image(image, size=(144, 144))
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs["pixel_values"], image

def post_processing(model_output, image, image_processor):
    custom_yolos_output = CustomYolosObjectDetectionOutput(
        logits=model_output[0],
        pred_boxes=model_output[1],
        last_hidden_state=model_output[2]
    )
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(custom_yolos_output, threshold=0.8, target_sizes=target_sizes)[0]
    return results

def single_frame_model_execution(input_path, model, labels_map, image_processor, output_path= "wow.jpg"):

    start_time = time.time()
    model_input, image = image_preprocessing(input_path, image_processor)
    end_time = time.time()
    print(f"Image PreProcessing: {end_time - start_time:.2f} seconds")
    
    # Pass the inputs to the model
    start_time = time.time()
    model_output = model(model_input)
    end_time = time.time()
    model_execution_time = end_time - start_time
    print(f"Model Execution: {end_time - start_time:.2f} seconds")

    # Image PostPorcessing
    start_time = time.time()
    results = post_processing(model_output, image, image_processor)
    end_time = time.time()
    print(f"Image PostProcessing: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    image_with_boxes = draw_boxes(image, results, labels_map)
    end_time = time.time()
    print(f"Draw boxes: {end_time - start_time:.2f} seconds")

    # Save the image with bounding boxes
    # start_time = time.time()
    # cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    # end_time = time.time()
    # print(f"Writing Image: {end_time - start_time:.2f} seconds")

    return image_with_boxes, model_execution_time

labels_map = {0: 'N/A', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 
                6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 
                11: 'fire hydrant', 12: 'N/A', 13: 'stop sign', 14: 'parking meter', 
                15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 
                21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'N/A', 
                27: 'backpack', 28: 'umbrella', 29: 'N/A', 30: 'N/A', 31: 'handbag', 
                32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 
                37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 
                41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 
                45: 'N/A', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 
                51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 
                56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 
                61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 
                66: 'N/A', 67: 'dining table', 68: 'N/A', 69: 'N/A', 70: 'toilet', 
                71: 'N/A', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 
                77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 
                82: 'refrigerator', 83: 'N/A', 84: 'book', 85: 'clock', 86: 'vase', 
                87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

# Load the model from the local file
model = torch.jit.load('yolos_tiny.torchscript')
model.eval()

image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

single_frame_model_execution("test.jpg", model, labels_map, image_processor, "result.jpg")
single_frame_model_execution("test.jpg", model, labels_map, image_processor, "result1.jpg")
single_frame_model_execution("test.jpg", model, labels_map, image_processor, "result2.jpg")
single_frame_model_execution("test.jpg", model, labels_map, image_processor, "result3.jpg")

input_dir = 'input_stream'
output_dir = 'output_stream'

ip_webcam_url = "http://192.168.18.5:8080/video"  # Replace with your IP webcam URL

cap = cv2.VideoCapture(ip_webcam_url)
frame_count = 0
total_model_execution_time = 0
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or (time.time() - start_time) > 10:
        break

    # Save the current frame in the input_stream folder
    input_path = os.path.join(input_dir, f'frame_no_{frame_count + 1}.jpg')
    cv2.imwrite(input_path, frame)

    # Run YOLO on the saved frame
    image_with_boxes, model_execution_time = single_frame_model_execution(input_path, model, labels_map, image_processor)
    total_model_execution_time += model_execution_time
    # Convert the processed image back to OpenCV format
    output_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)

    # Save the output image in the output_stream folder
    output_path = os.path.join(output_dir, f'frame_no_{frame_count + 1}.jpg')
    cv2.imwrite(output_path, output_image)

    # Display the frame with bounding boxes
    cv2.imshow('YOLO Object Detection', output_image)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update frame count
    frame_count += 1

end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time
print(f"Total Frames: {frame_count}")
print(f"Frame Rate: {fps:.2f} frames per second")
print("Total model execution time:", total_model_execution_time)
print("Avg model execution time per frame:", total_model_execution_time/ frame_count)
# Print the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

cap.release()
cv2.destroyAllWindows()