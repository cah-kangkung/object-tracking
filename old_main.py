import cv2 as cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='fish_tank.mp4')
parser.add_argument('--ke', type=str, help='Erosion Kernel Size.', default='5')
parser.add_argument('--kd', type=str, help='Dilation Kernel Size.', default='11')
parser.add_argument('--a', type=str, help='Learning Rate.', default='-1')
args = parser.parse_args()
  
resize_value = 2
  
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
    
# GMM BS #
backSub01 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    
kernel_erosion = np.ones((int(args.ke), int(args.ke)), np.uint8)
kernel_dilation = np.ones((int(args.kd), int(args.kd)), np.uint8)

count = 0

# loop through each frame    
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    height, width, layers = frame.shape
    new_h = height / resize_value
    new_w = width / resize_value
    frame = cv2.resize(frame, (int(new_w), int(new_h)))
    
    fg = backSub01.apply(frame, learningRate = int(args.a))
    
    gaussian_blur = cv2.GaussianBlur(frame,(5,5),0)
    fg_gaussian= backSub01.apply(gaussian_blur, learningRate = int(args.a))
    
    erosion = cv2.erode(fg_gaussian, kernel_erosion, iterations = 1)
    dilation = cv2.dilate(erosion, kernel_dilation, iterations = 1)
    
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        cv2.rectangle(dilation, (x, y), (x+w, y+h), (0,255,255), 2)
        
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    cv2.imshow('Frame', frame)
    #cv.imshow('GMM', fg)
    #cv.imshow('GMM with Gaussian Blurr', fg_gaussian)
    #cv.imshow('GMM after Erosion', erosion)
    cv2.imshow('GMM after Dialtion and Erosion', dilation)
    
    keyboard = cv2.waitKey(0)
    if keyboard == 27:
        break
    
    count += 1
  
capture.release()
cv2.destroyAllWindows()