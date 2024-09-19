# Detect objects using yolov8 and open cv

import cv2
import pandas as pd
import numpy as np

from ultralytics import YOLO

from tracker_abdo2 import *
#Define the pretrained model
model=YOLO("yolov8n.pt")

def getPosition(event, x,y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle where clicked
        cv2.imshow("Abdo-tracking", frame)
        print(x,y)
  
cv2.namedWindow("Abdo-tracking")
cv2.setMouseCallback("Abdo-tracking", getPosition)

#define classes
myFile = open("coco_classes.txt","r")
classNames = myFile.read().split("\n")
#print(classList)

# #Define the video source
#cap = cv2.VideoCapture("video2.mp4")

##Define the area
##area=[(296, 223),(356, 163),(719, 168),(773, 227)]

#points = np.array([[297, 222], [354, 163], [719, 168], [773, 227]], np.int32)
#points = points.reshape((-1, 1, 2))

# #img = cv2.imread("image.jpg")


# #Define the video source
cap = cv2.VideoCapture("video1.mp4")
##Define the area
##area=[(270, 238),(294, 280),(592, 226),(552, 207)]
##right road
points1 = np.array([[443, 239],[575, 220],[612, 240],[475, 265]], np.int32)
points1 = points1.reshape((-1, 1, 2))

##left road
points2 = np.array([[270, 238], [403, 223], [436, 255], [294, 281]], np.int32)
points2 = points2.reshape((-1, 1, 2))


tracker = Tracker()
count=0

# Initialize sets for each category for the left road
counter_car_left = set()
counter_truck_left = set()
counter_bus_left = set()
counter_moto_left= set()
counter_bike_left = set()

# Initialize sets for each category for the right road
counter_car_right = set()
counter_truck_right = set()
counter_bus_right = set()
counter_moto_right= set()
counter_bike_right = set()

# Create a mapping from labels to the corresponding sets for the left road
category_map_left = {
    "car": counter_car_left,
    "truck": counter_truck_left,
    "bus": counter_bus_left,
    "motorcycle": counter_moto_left,
    "bike": counter_bike_left
}

# Create a mapping from labels to the corresponding sets for the right road
category_map_right = {
    "car": counter_car_right,
    "truck": counter_truck_right,
    "bus": counter_bus_right,
    "motorcycle": counter_moto_right,
    "bike": counter_bike_right
}

# Assign different colors to each category in BGR format
color_map = {
    "car": (255, 0, 0), 
    "truck": (0, 255, 0), 
    "bus": (0, 0, 255), 
    "motorcycle": (255, 0, 255),
    "bike": (255, 255, 0)    
}


width = 1020
height = 500

# Define starting position and text parameters
x_right_position = width - 200
x_left_position = 20
y_position = 90  # Starting y-position for text
line_height = 30  # Distance between each line of text

#font = cv2.FONT_HERSHEY_PLAIN
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2
padding = 60
#------------------------------------------

while True:
    success,frame=cap.read()
    
    if success:
     
        frame=cv2.resize(frame,(width,height))
        results = model.predict(frame, verbose=False)
        

        # Extract bounding boxes, classes, and confidences
        # results[0].boxes.data contains all the necessary information
        boxes_data = results[0].boxes.data.numpy()  # Convert tensor to numpy array

        # Create a DataFrame from the data
        # The columns could be x_min, y_min, x_max, y_max, confidence, class
        df = pd.DataFrame(boxes_data, columns=["x1", "y1", "x2", "y2", "confidence", "class"])

#         # You can now perform operations like filtering based on confidence, etc.
#         filtered_df = df[df['confidence'] > 0.5]
#         print(filtered_df)
#         print("*********************************")

        list=[]
        # Iterate over each row in the DataFrame to draw rectangles
        for index, row in df.iterrows():
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            confidence = row['confidence']
            class_id = int(row['class'])
 
            if(confidence>0.50):   # .6 worked ok except one truck
                    label = classNames[class_id]
                    if label in ['car','truck','bus','motorcycle','bike']:
                        list.append([x1, y1, x2, y2,label])

        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x1, y1, x2, y2,label,id =bbox
         
            # get the center point
            cx=(x1+x2)//2
            cy=(y1+y2)//2
            
            state = cv2.pointPolygonTest(points1,(cx,cy),False)
            if state == 1:
                # Check if the label exists in the dictionary and add to the corresponding set
                if label in category_map_right:
                    category_map_right[label].add(id)
            else:
                 state = cv2.pointPolygonTest(points2,(cx,cy),False)
                 if state == 1:
                    # Check if the label exists in the dictionary and add to the corresponding set
                    if label in category_map_left:
                        category_map_left[label].add(id)
                 else:
                        continue
            color = color_map.get(label)
            # Draw a rectangle on the image
            #cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # Green box with thickness 2
                  
            # # Draw a circle at the center
            cv2.circle(frame,(cx,cy),4,color,-1)

            # Put class ID and confidence on the box
            cv2.putText(frame, str(id), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Blue text
        
        # Loop through each category and display the count
        for i, (category, counter) in enumerate(category_map_right.items()):
            color =color_map.get(category)
            if category == "motorcycle":
                category="moto"           
            # Format the text with left-aligned category and right-aligned count
            text1 = f"{category:<6}"  # Class to display
            cv2.putText(frame, text1, (x_right_position, y_position + i * line_height), font, font_scale, color, thickness)
            text2 = f"{len(counter):>5}"  # Class count   font, font_scale,  color, thickness
            cv2.putText(frame, text2, (x_right_position + padding, y_position + i * line_height), font, font_scale, color, thickness)
   
        # Loop through each category and display the count
        for i, (category, counter) in enumerate(category_map_left.items()):
            color =color_map.get(category)
            if category == "motorcycle" :
                category="moto"
                
            text1 = f"{category:<6}"  # Class to display
            cv2.putText(frame, text1, (x_left_position, y_position + i * line_height), font, font_scale, color, thickness)
            text2 = f"{len(counter):>5}"  # Class count   font, font_scale,  color, thickness
            cv2.putText(frame, text2, (x_left_position + padding, y_position + i * line_height), font, font_scale, color, thickness)
       
         # Draw the polygon
        cv2.polylines(frame, [points1], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [points2], isClosed=True, color=(255, 0, 0), thickness=2)
        
        cv2.imshow("Abdo-tracking", frame)

    else:
        break
    if cv2.waitKey(1)&0xFF==27:
        break
    
    
cap.release()
cv2.destroyAllWindows()
