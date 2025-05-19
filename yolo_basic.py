from ultralytics import YOLO
import numpy

#loding a pretrained model
# Load a model
model = YOLO("yolo11n.pt")


detection_output = model.predict(source="testfiles/images/desk.jpg",conf=0.25,save=True) #give ur image path so that it detects n then put "save = true" so that it saves the results in "runs" folder

#display tensor array
print (detection_output)

#Display numpy array
print (detection_output[0].numpy())