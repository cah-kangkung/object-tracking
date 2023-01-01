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
    video_path = "datasets/f4k_detection_tracking/gt_124.flv"
    kernel_erosion_size = 3
    kernel_dilation_size = 13
    scale = 4
    downsampling_mode = "resize"
    max_frame_to_skip = 10

    capture = cv2.VideoCapture(video_path)
    detector = Detector(kernel_erosion_size, kernel_dilation_size, scale, downsampling_mode)
    tracker = Tracker(max_frame_to_skip)
    
    skiped_frame = 50
    frame_count = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frame_count += 1

        detections, contours, morphed = detector.detect(frame)
        
        if frame_count > skiped_frame:
            tracker.update_tracks(detections, frame)

        # Frame
        cv2.rectangle(frame, (10, 2), (80, 20), (255, 255, 255), -1)
        cv2.putText(
            frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (13, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
        )

        # Count
        cv2.rectangle(frame, (100, 2), (130, 20), (255, 255, 255), -1)
        cv2.putText(
            frame, str(len(tracker.tracks)), (105, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
        )

        # Count
        cv2.rectangle(frame, (150, 2), (400, 20), (255, 255, 255), -1)
        cv2.rectangle(frame, (155, 6), (175, 16), (255, 0, 0), -1)
        cv2.putText(
            frame, '=current', (180, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
        )
        cv2.rectangle(frame, (270, 6), (290, 16), (0, 0, 255), -1)
        cv2.putText(
            frame, '=KF', (295, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
        )

        cv2.imshow("Frame", frame)

        keyboard = cv2.waitKey(0)
        if keyboard == 27:
            break
        elif keyboard == ord("s"):
            cv2.imwrite("screenshots/morphed_frame_" + str(capture.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", morphed)
            cv2.imwrite("screenshots/original_frame_" + str(capture.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", frame)
        
    capture.release()
    cv2.destroyAllWindows()

def downsampling():
    file_name = "morphed_frame_30.0"
    image = cv2.imread("screenshots/" + file_name + ".jpg")
    cv2.imshow("Original Image", image)

    scale = 8
    method = "resize"
    detector = Detector(1, 1, scale, method)
    contour_tracing = ContourTracing()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    downsampled_image = detector.downsample(gray, scale, method)

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
        x, y, w, h = cv2.boundingRect(contour)
        upscale_x = x * scale
        upscale_y = y * scale
        upscale_w = w * scale
        upscale_h = h * scale
        cX = x + w // 2
        cY = y + h // 2
        upscale_cX = upscale_x + upscale_w // 2
        upscale_cY = upscale_y + upscale_h // 2

        detection = np.array([[cX], [cY], [w], [h]])
        detections.append(np.round(detection))

        cv2.rectangle(image, (upscale_x, upscale_y), (upscale_x + upscale_w, upscale_y + upscale_h), (0, 255, 255), 2)
        cv2.circle(downsampled_image, (cX, cY), 1 // scale, (0, 0, 255), -1)
        cv2.circle(image, (upscale_cX, upscale_cY), 3, (255, 0, 0), -1)


    string =  "scale: x" + str(scale) + ", count: " + str(len(contours_second)) + ", time: " + str(end_time_second - start_time_second)
    cv2.putText(
        image, string, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255)
    )

    cv2.imshow("Downsampled Image", downsampled_image)
    cv2.imshow("Traced Image", image)

    if method == "pyr":
        cv2.imwrite("screenshots/" + file_name + "_x" + str(scale) + "_downsampled_at.jpg", downsampled_image)
        cv2.imwrite("screenshots/" + file_name + "_x" + str(scale) + "_original_at.jpg", image)
    else:    
        cv2.imwrite("screenshots/" + file_name + "_x" + str(scale) + "_resize_downsampled_at.jpg", downsampled_image)
        cv2.imwrite("screenshots/" + file_name + "_x" + str(scale) + "_resize_original_at.jpg", image)


    cv2.waitKey(0)    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    downsampling()
