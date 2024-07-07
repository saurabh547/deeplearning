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
