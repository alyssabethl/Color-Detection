import numpy as np
import cv2

# capture video through webcam
webcam = cv2.VideoCapture(0)

while(1):
  # read video in image frames
  _, imageFrame = webcam.read()

  # convert imageFrame from RGB to HSV color space
  hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

  # set red color range and define mask
  red_lower = np.array([136, 87, 111], np.uint8)
  red_upper = np.array([180, 255, 255], np.uint8)
  red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

  # set green color range and define mask
  green_lower = np.array([25, 52, 72], np.uint8)
  green_upper = np.array([102, 255, 255], np.uint8)
  green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

  # set blue color range and define mask
  blue_lower = np.array([94, 80, 2], np.uint8)
  blue_upper = np.array([120, 255, 255], np.uint8)
  blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

  # set black color range and define mask
  black_lower = np.array([0, 0, 0], np.uint8)
  black_upper = np.array([64, 64, 64], np.uint8)
  black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

  # Morphological Transformation: Dilation to remove image noise
  # for each color and bitwise_and operator between imageFrame and mask
  # is determined to detect a specific color
  kernel = np.ones((5, 5), "uint8")

  red_mask = cv2.dilate(red_mask, kernel)
  red_red = cv2.bitwise_and(imageFrame, imageFrame, mask = red_mask)

  green_mask = cv2.dilate(green_mask, kernel)
  green_green = cv2.bitwise_and(imageFrame, imageFrame, mask = green_mask)

  blue_mask = cv2.dilate(blue_mask, kernel)
  blue_blue = cv2.bitwise_and(imageFrame, imageFrame, mask = blue_mask)

  black_mask = cv2.dilate(black_mask, kernel)
  black_black = cv2.bitwise_and(imageFrame, imageFrame, mask = black_mask)

  # create contour to track red
  contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if(area > 300):
      x, y, w, h = cv2.boundingRect(contour)
      imageFrame = cv2.rectangle(imageFrame, (x, y), (x+w, y+h),
                                 (0, 0, 0), 3)
      
      cv2.putText(imageFrame, "RED", (x,y), 
                  cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
      
  # create contour to track green
  contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if(area > 300):
      x, y, w, h = cv2.boundingRect(contour)
      imageFrame = cv2.rectangle(imageFrame, (x, y), (x+w, y+h),
                                 (0, 0, 0), 3)
      
      cv2.putText(imageFrame, "GREEN", (x,y), 
                  cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
                
  # create contour to track blue
  contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if(area > 300):
      x, y, w, h = cv2.boundingRect(contour)
      imageFrame = cv2.rectangle(imageFrame, (x, y), (x+w, y+h),
                                 (0, 0, 0), 3)
      
      cv2.putText(imageFrame, "BLUE", (x,y), 
                  cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
      
  # create contour to track black
  contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if(area > 300):
      x, y, w, h = cv2.boundingRect(contour)
      imageFrame = cv2.rectangle(imageFrame, (x, y), (x+w, y+h),
                                 (0, 0, 0), 3)
      
      cv2.putText(imageFrame, "BLACK", (x,y), 
                  cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
      
  # program termination
  cv2.imshow("Color Detection", imageFrame)
  if cv2.waitKey(10) & 0xFF == ord('q'):
    webcam.release()
    cv2.destroyAllWindows()
    break
  
