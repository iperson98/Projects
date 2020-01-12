import imutils
from imutils.video import FPS
import numpy as np
import argparse
import time
import cv2

# Parsing command line arguments that map to the trained network
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True,
    help = "Path to prototext file.")
ap.add_argument("-m", "--model", required = True,
    help = "Path to pre-trained model")
ap.add_argument("-c", "--confidence", type = float, default = 0.2,
    help = "Minimum probability to filter unlikely detections")
args = vars(ap.parse_args())

#Classes defined in our trained neural network
CLASSES = ["background", "airplane", "bike", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "monitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#load out pre trained model from usb/other
print("[INFO] loading model...")
nnet = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#initalize video stream to be viewed by user
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()

#After initialzing all our parameters, we can now loop over all frames 
#incoming from the Pi camera module
while True:
    
    #Reading the frame from the video stream and resizing it
    #to a width of 400 pixels -- as it reduces computational intensity
    video_check, frame = vs.read()
    #frame = imutils.resize(frame, width = 400)
    
    #Reformating the frames into a "blob" that can be easily understood by
    #the trained neural network
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    
    #Runnign the blob through the neural network                           
    nnet.setInput(blob)
    detections = nnet.forward()
    
    #For the total number of detections found in our reformated frame
    for i in np.arange(0, detections.shape[2]):
        
        #Extract how confident we are in the detection
        confidence = detections[0, 0, i, 2]
        
        #If the confidence is not greater than our threshold (default to 0.2)
        #then we do not display the bounding box
        if confidence > args["confidence"]:
            #Parameters for creating bounding box
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            #Draw what the neural network thinks is in the box
            label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startY, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        #Finally we output the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        #break from the loop if q is pressed
        if key == ord("q"):
            break
#update the fps counter
fps.update()
        
#Stop the timer and display all relevent parameters for testing purposes
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approximate FPS: {:.2f}".format(fps.fps()))

#Close all windowns and the video stream
cv2.destroyAllWindows()
vs.stop()