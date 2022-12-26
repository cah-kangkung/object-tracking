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
    video_path = "datasets/fish_tank_02.mp4"
    kernel_erosion_size = 5
    kernel_dilation_size = 7
    scale = 8
    downsampling_mode = "resize"

    capture = cv2.VideoCapture(video_path)
    detector = Detector(kernel_erosion_size, kernel_dilation_size, scale, downsampling_mode)
    tracker = Tracker(20)
    skiped_frame = 50

    frame_count = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frame_count += 1

        detections, contours = detector.detect(frame)
        
        if frame_count > skiped_frame:
            tracker.update_tracks(detections, frame)

        cv2.rectangle(frame, (10, 2), (80, 20), (255, 255, 255), -1)
        cv2.putText(
            frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
        )
        cv2.rectangle(frame, (100, 2), (130, 20), (255, 255, 255), -1)
        cv2.putText(
            frame, str(len(tracker.tracks)), (105, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
        )
        cv2.imshow("Frame", frame)

        keyboard = cv2.waitKey(50)
        if keyboard == 27:
            break
        elif keyboard == ord("s"):
            # cv2.imwrite("screenshots/threshold_frame_" + str(capture.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", thresh)
            cv2.imwrite("screenshots/original_frame_" + str(capture.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", frame)
        
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
