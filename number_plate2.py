import cv2
import numpy as np

import pytesseract as pyt

harcascade = "model/haarcascade_russian_plate_number.xml"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

cap.set(3, 640) # width
cap.set(4, 480) #height

min_area = 500
count = 0

while True:
  success, img = cap.read()

  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  plate_cascade = cv2.CascadeClassifier(harcascade)
  plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
  faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
  
  for (x,y,w,h) in faces:
	  cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 5)
	  roi_gray=img_gray[y:y+w, x:x+w]
	  roi_color=img[y:y+h, x:x+w]
  for (x,y,w,h) in plates:
      area = w * h
      if area > min_area:
          cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
          cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
          img_roi = img[y: y+h, x:x+w]
          cv2.imshow("ROI", img_roi)
  
  cv2.imshow("Result", img)
  if cv2.waitKey(1) & 0xFF == ord('s'):
      cv2.imwrite("plates/img" + ".jpg", img_roi)
      ###############
      results=pyt.image_to_string("plates/img.jpg", lang="eng")
      print(results)
      cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
      cv2.putText(img, results, (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
      cv2.imshow("Results",img)
      cv2.waitKey(500)