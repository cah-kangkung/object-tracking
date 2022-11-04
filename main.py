import copy as copy
from pprint import pprint
from sqlite3 import adapters
import cv2 as cv2
from detector import Detector
from tracker import Tracker
from contour_tracing import ContourTracing
import numpy as np
import time 

def main():
    # Input
    video_path = "datasets/fish_tank_04.mp4"
    kernel_erosion_size = 5
    kernel_dilation_size = 7
    resize_value = 1

    capture = cv2.VideoCapture(video_path)
    detector = Detector(kernel_erosion_size, kernel_dilation_size)
    tracker = Tracker()
    skiped_frame = 50

    frame_count = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frame_count += 1

        height, width, layers = frame.shape
        new_h = height / resize_value
        new_w = width / resize_value
        frame = cv2.resize(frame, (int(new_w), int(new_h)))

        detections, contours = detector.detect(frame)
        if frame_count > skiped_frame:
            tracker.update_tracks(detections, frame)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(
            frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
        )
        cv2.imshow("Frame", frame)

        keyboard = cv2.waitKey(0)
        if keyboard == 27:
            break
        elif keyboard == ord("s"):
            # cv2.imwrite("screenshots/threshold_frame_" + str(capture.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", thresh)
            cv2.imwrite("screenshots/original_frame_" + str(capture.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", frame)

    capture.release()
    cv2.destroyAllWindows()

def downsampling():
    image = cv2.imread("screenshots/original_frame_39.0.jpg")
    detector = Detector(1, 1)
    detector.downsample(image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour_tracing():
    image = cv2.imread("datasets/sample_01.jpg")
    cv2.imshow("Original Image", image)

    scale = 2
    detector = Detector(1, 1)
    contour_tracing = ContourTracing()

    downsampled_image = detector.downsample(image, scale)
    cv2.imshow("Downsampled Image", downsampled_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    start_time_first = time.time()
    # contours_first, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    contours_first = contour_tracing.findCountourCustom(gray) 
    end_time_first = time.time()

    start_time_second = time.time()
    # contours_second, _ = cv2.findContours(downsampled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_second = contour_tracing.findCountourCustom(downsampled_image)
    end_time_second = time.time()


    print("Original Image", end_time_first - start_time_first)
    print("Downsampled Image", end_time_second - start_time_second)
    print("Jumlah", len(contours_second))

    # Find centers, width, and height for each contour / object
    detections = []
    for contour in contours_second:
        # print("CONTOUR")
        # print(contour.shape)
        # print(contour)
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"]) * scale
        cY = int(M["m01"] / M["m00"]) * scale
        x, y, w, h = cv2.boundingRect(contour)
        x = x * scale
        y = y * scale
        w = w * scale
        h = h * scale

        detection = np.array([[cX], [cY], [w], [h]])
        detections.append(np.round(detection))

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 3, (255, 0, 0), -1)

    cv2.putText(
        image, "time: " + str(end_time_second - start_time_second), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255)
    )

    cv2.imshow("Traced Image", image)

    cv2.imwrite("screenshots/center_01_x" + str(scale) + ".jpg", image)
    
    cv2.waitKey(0)    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    contour_tracing()
