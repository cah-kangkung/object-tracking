import cv2 as cv2
from cv2 import log
import numpy as np
from contour_tracing import ContourTracing
import time 

class Detector(object):
    def __init__(self, kernel_erosion_size, kernel_dilation_size, scale):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.contour_tracing = ContourTracing()
        self.kernel_erosion_size = kernel_erosion_size
        self.kernel_dilation_size = kernel_dilation_size
        self.scale = scale

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Subtract bg using GMM
        fg_mask = self.bg_subtractor.apply(gray)

        # Remove noise using morphology
        kernel_erosion = np.ones((self.kernel_erosion_size, self.kernel_erosion_size), np.uint8)
        kernel_dilation = np.ones((self.kernel_dilation_size, self.kernel_dilation_size), np.uint8)
        morphed = self.morph(fg_mask, kernel_erosion, kernel_dilation)

        # Downsample
        downsampled_image = self.downsample(morphed, self.scale)

        start_time_first = time.time()
        # Find contours
        contours = self.contour_tracing.findCountourCustom(downsampled_image)
        # contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        end_time_first = time.time()

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

        print(end_time_first - start_time_first)

        return detections, contours

    def morph(self, fg_mask, kernel_erosion, kernel_dilation):
        erosion = cv2.erode(fg_mask, kernel_erosion, iterations=1)
        # cv2.imshow("Erosion", erosion)
        dilation = cv2.dilate(erosion, kernel_dilation, iterations=1)
        # cv2.imshow("Dilation", dilation)

        return dilation

    def downsample(self, frame, scale):
        
        times = np.sqrt(scale)

        # Using averaging each scale pixels into 1 pixel
        # downsampled_image = np.zeros((new_rows, new_cols))
        # blurred_image = cv2.GaussianBlur(frame, (5,5) ,0)
        # for i in range(0, rows - scale, scale):
        #     for j in range(0, cols - scale, scale):
        #         new_i = i // scale
        #         new_j = j // scale
        #         downsampled_image[new_i][new_j] = np.mean(blurred_image[i:i + scale, j:j + scale])

        # downsampled_image = downsampled_image.astype(np.uint8)

        for i in range(int(np.round(times))):
            rows, cols = frame.shape
            frame = cv2.pyrDown(frame, dstsize=(cols // 2, rows // 2 ))
        
        return frame

