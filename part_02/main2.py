# Detect objects using yolov8 and open cv
import cv2
import pandas as pd
import numpy as np

from ultralytics import YOLO

from tracker_abdo2 import *

#Define the pretrained model
model=YOLO("yolov8n.pt")

## Draw a circle whith mouse click
def getPosition(event, x,y, flags, param):
      # Left mouse button click
  if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a green circle where clicked
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  
        cv2.imshow("Abdo-tracking", frame)
        print(x,y)
  
cv2.namedWindow("Abdo-tracking")
cv2.setMouseCallback("Abdo-tracking", getPosition)

#define classes
myFile = open("coco_classes.txt","r")
classNames = myFile.read().split("\n")
#print(classList)

## video1
# cap = cv2.VideoCapture("video1.mp4")
# points = np.array([[270, 238], [294, 280], [592, 226], [552, 207]], np.int32)
# points = points.reshape((-1, 1, 2))


## video2
cap = cv2.VideoCapture("video2.mp4")
points = np.array([[297, 222], [354, 163], [719, 168], [773, 227]], np.int32)
points = points.reshape((-1, 1, 2))


tracker = Tracker()

# Initialize sets for each category
counter_car = set()
counter_truck = set()
counter_bus = set()
counter_bike = set()

# Create a mapping from labels to the corresponding sets
category_map = {
    "car": counter_car,
    "truck": counter_truck,
    "bus": counter_bus,
    "bike": counter_bike
}

# Assign different colors to each category in BGR format
color_map = {
    "car": (255, 0, 0),      # Blue for Car
    "truck": (0, 255, 0),    # Green for Truck
    "bus": (0, 0, 255),      # Red for Bus
    "bike": (255, 255, 0)    # Cyan for Bike
}

# Define text starting position and parameters
x_position = 850
y_position = 100
line_height = 30  

#------------------------------------------

while True:
    success,frame=cap.read()
    
    if success:
        frame=cv2.resize(frame,(1020,500))
        results = model.predict(frame,verbose=False)
 
        # Extract bounding boxes, classes, and confidences
        # results[0].boxes.data contains all the necessary information
        boxes_data = results[0].boxes.data.numpy()  # Convert tensor to numpy array

        # Create a DataFrame from the data
        # The columns could be x_min, y_min, x_max, y_max, confidence, class
        df = pd.DataFrame(boxes_data, columns=["x1", "y1", "x2", "y2", "confidence", "class"])

#--------------------------------------
        list=[]
        # Iterate over each row in the DataFrame to draw rectangles
        for index, row in df.iterrows():
            print(index)
            print(row)
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            confidence = row['confidence']
            class_id = int(row['class'])
 
            if(confidence>0.5):   # .6 worked ok except one truck
                    label = classNames[class_id]
                    if label in ['car','truck','bus','motorcycle']:
                        list.append([x1, y1, x2, y2,label])
 
        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x1, y1, x2, y2,label,id =bbox
         
            # get the center point
            cx=(x1+x2)//2
            cy=(y1+y2)//2

            color = color_map.get(label)
             # Draw a rectangle on the image
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # Green box with thickness 2
                  
            # Draw a circle at the center
            cv2.circle(frame,(cx,cy),4,color,-1)

            # Put class ID and confidence on the box
            cv2.putText(frame, str(id), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  # Blue text           
          
            
            state = cv2.pointPolygonTest(points,(cx,cy),False)
            if state == 1:
               
                 # Check if the label exists in the dictionary and add to the corresponding set
                if label in category_map:
                    category_map[label].add(id)
                
        # Loop through each category and display the count
        for i, (category, counter) in enumerate(category_map.items()):
            text = f"{category}: {len(counter)}"  # Text to display
            #color = color_map[category]  # Get the color for the current category
            color =color_map.get(category)
            cv2.putText(frame, text, (x_position, y_position + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
       
        ## Draw the area
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        cv2.imshow("Abdo-tracking", frame)

    else:
        break
    if cv2.waitKey(1)&0xFF==27:
        break
    
    
cap.release()
cv2.destroyAllWindows()
