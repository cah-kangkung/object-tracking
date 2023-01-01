import cv2 as cv2
from cv2 import log
import numpy as np
from contour_tracing import ContourTracing
import time 

class Detector(object):
    def __init__(self, kernel_erosion_size, kernel_dilation_size, scale, downsampling_mode):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.contour_tracing = ContourTracing()
        self.kernel_erosion_size = kernel_erosion_size
        self.kernel_dilation_size = kernel_dilation_size
        self.scale = scale
        self.downsampling_mode = downsampling_mode

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Subtract bg using GMM
        fg_mask = self.bg_subtractor.apply(gray)
        cv2.imshow("GMM", fg_mask)

        # Remove noise using morphology
        kernel_erosion = np.ones((self.kernel_erosion_size, self.kernel_erosion_size), np.uint8)
        kernel_dilation = np.ones((self.kernel_dilation_size, self.kernel_dilation_size), np.uint8)
        morphed = self.morph(fg_mask, kernel_erosion, kernel_dilation)

        # Downsample
        downsampled_image = self.downsample(morphed, self.scale, self.downsampling_mode)
        
        # Find contours
        contours = self.contour_tracing.findCountourCustom(downsampled_image)

        # Find centers, width, and height for each contour / object
        # Rescale to original size
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x = x * self.scale
            y = y * self.scale
            w = w * self.scale
            h = h * self.scale
            cX = x + w // 2
            cY = y + h // 2

            detection = np.array([[cX], [cY], [w], [h]])
            detections.append(np.round(detection))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(frame, (cX, cY), 3, (255, 0, 0), -1)

        return detections, contours, morphed

    def morph(self, fg_mask, kernel_erosion, kernel_dilation):
        erosion = cv2.erode(fg_mask, kernel_erosion, iterations=1)
        cv2.imshow("Erosion", erosion)
        dilation = cv2.dilate(erosion, kernel_dilation, iterations=1)
        cv2.imshow("Dilation", dilation)

        return dilation

    def downsample(self, frame, scale, mode="resize"):
        
        times = np.sqrt(scale)

        for i in range(int(np.round(times))):
            rows, cols = frame.shape
            if mode == "pyr":
                frame = cv2.pyrDown(frame, dstsize=(cols // 2, rows // 2 ))
            elif mode == "resize":
                frame = cv2.resize(frame, dsize=(cols // 2, rows // 2 ))
        
        return frame

