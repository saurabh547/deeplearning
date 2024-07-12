#pip install numpy opencv-python opencv-python-headless

import cv2

# Load the image
image = cv2.imread(r'C:\Users\kaniya12\Desktop\practicals\horse.jpg')
image = cv2.resize(image, (640, 480))
h, w = image.shape[:2]

# Path to the weights and model files
weights = r'C:\Users\kaniya12\Desktop\practicals\8practicalImageVideo\frozen_inference_graph.pb'
model = r'C:\Users\kaniya12\Desktop\practicals\8practicalImageVideo\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Load the MobileNet SSD model trained on the COCO dataset
net = cv2.dnn.readNetFromTensorflow(weights, model)

# Load the class labels the model was trained on
class_names = []
with open(r'C:\Users\kaniya12\Desktop\practicals\8practicalImageVideo\coco-labels-paper.txt', 'r') as f:
    class_names = f.read().strip().split("\n")

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0/127.5, (320, 320), [127.5, 127.5, 127.5])

# Pass the blob through the network and get the output predictions
net.setInput(blob)
output = net.forward()

# Loop over the number of detected objects
for detection in output[0, 0, :, :]:
    probability = detection[2]

    # If the confidence of the model is lower than 50%, skip it
    if probability < 0.5:
        continue

    # Perform element-wise multiplication to get the (x, y) coordinates of the bounding box
    box = [int(a * b) for a, b in zip(detection[3:7], [w, h, w, h])]
    box = tuple(box)

    # Draw the bounding box of the object
    cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), thickness=2)

    # Extract the ID of the detected object to get its name
    class_id = int(detection[1])

    # Draw the name of the predicted object along with the probability
    label = f"{class_names[class_id - 1].upper()} {probability * 100:.2f}%"
    cv2.putText(image, label, (box[0], box[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the output image with detected objects
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# video detection
import numpy as np
import cv2
import datetime

# Video capture
video_cap = cv2.VideoCapture("examples/video1.mp4")

# Grab the width and height of the video stream
frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap.get(cv2.CAP_PROP_FPS))

# Initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# Path to the weights and model files
weights = "ssd_mobilenet/frozen_inference_graph.pb"
model = "ssd_mobilenet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# Load the MobileNet SSD model trained on the COCO dataset
net = cv2.dnn.readNetFromTensorflow(weights, model)

# Load the class labels the model was trained on
class_names = []
with open("ssd_mobilenet/coco_names.txt", "r") as f:
    class_names = f.read().strip().split("\n")

# Create a list of random colors to represent each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')

# Loop over the frames
while True:
    # Start time to compute the fps
    start = datetime.datetime.now()
    success, frame = video_cap.read()
    if not success:
        break

    h, w = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (320, 320), [127.5, 127.5, 127.5])

    # Pass the blob through our network and get the output predictions
    net.setInput(blob)
    output = net.forward()

    # Loop over the number of detected objects
    for detection in output[0, 0, :, :]:
        # The confidence of the model regarding the detected object
        probability = detection[2]
        if probability < 0.5:
            continue

        # Extract the ID of the detected object to get its name and the color associated with it
        class_id = int(detection[1])
        label = class_names[class_id - 1].upper()
        color = colors[class_id % len(colors)].tolist()

        # Perform element-wise multiplication to get the (x, y) coordinates of the bounding box
        box = [int(a * b) for a, b in zip(detection[3:7], [w, h, w, h])]
        (startX, startY, endX, endY) = box

        # Draw the bounding box of the object
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Draw the name of the predicted object along with the probability
        text = f"{label} {probability * 100:.2f}%"
        cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # End time to compute the fps
    end = datetime.datetime.now()
    fps_text = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the output frame
    cv2.imshow("Output", frame)
    
    # Write the frame to disk
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture, video writer, and close all windows
video_cap.release()
writer.release()
cv2.destroyAllWindows()

