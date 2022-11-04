import cv2 as cv2
from cv2 import log
import numpy as np
from contour_tracing import ContourTracing


class Detector(object):
    def __init__(self, kernel_erosion_size, kernel_dilation_size):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.contour_tracing = ContourTracing()
        self.kernel_erosion_size = 3
        self.kernel_dilation_size = 7

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Subtract bg using GMM
        fg_mask = self.bg_subtractor.apply(gray)

        # Remove noise using morphology
        kernel_erosion = np.ones((self.kernel_erosion_size, self.kernel_erosion_size), np.uint8)
        kernel_dilation = np.ones((self.kernel_dilation_size, self.kernel_dilation_size), np.uint8)
        morphed = self.morph(fg_mask, kernel_erosion, kernel_dilation)

        # Thresholding
        # ret, thresh = cv2.threshold(morphed, 0, 255, 0)
        # cv2.imshow("Thresholded", thresh)

        # Find contours
        # contours = self.contour_tracing.findCountourCustom(thresh)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Find centers, width, and height for each contour / object
        detections = []
        for contour in contours:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(contour)

            detection = np.array([[cX], [cY], [w], [h]])
            detections.append(np.round(detection))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        return detections, contours

    def morph(self, fg_mask, kernel_erosion, kernel_dilation):
        erosion = cv2.erode(fg_mask, kernel_erosion, iterations=1)
        cv2.imshow("Erosion", erosion)
        dilation = cv2.dilate(erosion, kernel_dilation, iterations=1)
        cv2.imshow("Dilation", dilation)

        return dilation

    def downsample(self, frame, scale):
        rows, cols, _ = frame.shape
        new_rows = rows // scale
        new_cols = cols // scale
        grayed_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Using opencv pyrdown which eliminate every event-numbered row and column
        # This method already applied gaussian blurr
        #downsampled_image = cv2.pyrDown(grayed_image) 

        # Using averaging each scale pixels into 1 pixel
        downsampled_image = np.zeros((new_rows, new_cols))
        blurred_image = cv2.GaussianBlur(grayed_image, (5,5) ,0)
        for i in range(0, rows - scale, scale):
            for j in range(0, cols - scale, scale):
                new_i = i // scale
                new_j = j // scale
                downsampled_image[new_i][new_j] = np.mean(blurred_image[i:i + scale, j:j + scale])

        downsampled_image = downsampled_image.astype(np.uint8)
        
        return downsampled_image

