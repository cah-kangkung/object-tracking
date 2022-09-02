import cv2 as cv2
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
        ret, thresh = cv2.threshold(morphed, 0, 255, 0)
        cv2.imshow("Thresholded", thresh)

        # Find contours
        contours = self.contour_tracing.findCountourCustom(thresh)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Find centers for each contour / object
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center_coordinate = np.array([[cX], [cY], [0], [0]])
            centers.append(np.round(center_coordinate))

        return centers, contours, thresh

    def morph(self, fg_mask, kernel_erosion, kernel_dilation):
        erosion = cv2.erode(fg_mask, kernel_erosion, iterations=1)
        cv2.imshow("Erosion", erosion)

        dilation = cv2.dilate(erosion, kernel_dilation, iterations=1)
        cv2.imshow("Dilation", dilation)

        return dilation


#

