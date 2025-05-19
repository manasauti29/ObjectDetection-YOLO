import random
import cv2
import numpy as np
from ultralytics import YOLO

#80 classes that are available in COCO dataset
class_list = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
              "scissors", "teddy bear", "hair drier", "toothbrush"]

# Function to generate a color gradient for class IDs
def get_gradient_color(class_id, num_classes=80):
    # Use HSV color space to create a gradient effect
    hsv_value = class_id / num_classes  # Normalizing class ID
    color = (hsv_value * 180, 255, 255)  # Use Hue, Full Saturation and Value
    bgr_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr_color)

# Generate colors for each class based on gradient
detection_colors = [get_gradient_color(i) for i in range(len(class_list))]

model = YOLO("weights/yolo11m.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 800
frame_hyt = 600

# Vals to resize video frames | small frame optimise the run
#frame_wid = 960
#frame_hyt = 540


cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("inference/videos/input.mp4") to take video as a input

if not cap.isOpened():
    print("Cannot open camera")
    exit()

window_width = 800
window_height = 600

# Create a resizable window
cv2.namedWindow("ObjectDetection", cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
cv2.resizeWindow("ObjectDetection", window_width, window_height)  # Set window size

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]


            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],  # Use color based on class ID
                3
            )


            # Display class name and confidence
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

#to release the capture
cap.release()
cv2.destroyAllWindows()