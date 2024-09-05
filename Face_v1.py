import numpy as np
import pandas as pd 
import matplotlib.pyplot as mlt
import seaborn as sns 
import tensorflow as tf
import sys
import cv2

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture= cv2.VideoCapture(0)

# Define the counter for naming screenshots
face_counter = 0

while True:
    #Capturing Frame by Frame
    ret, frame =video_capture.read()
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces= faceCascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    
    )

    # Rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]

        # Save the detected face image
        face_filename = f"face_{face_counter}.jpg"
        cv2.imwrite(face_filename, face)
        face_counter += 1

        # Analyze the captured face image (placeholder)
        # You can replace this with your actual analysis code
        print(f"Captured and saved face as {face_filename}")

        # Optional: Display the cropped face
        cv2.imshow('Captured Face', face)
        cv2.waitKey(1000)  # Display the captured face for 1 second

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()