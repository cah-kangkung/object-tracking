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

    capture = cv2.VideoCapture(video_path)
    detector = Detector(kernel_erosion_size, kernel_dilation_size, scale)
    tracker = Tracker()
    skiped_frame = 50

    frame_count = 0
    while True:
        start_time_first = time.time()
        ret, frame = capture.read()
        if frame is None:
            break
        frame_count += 1

        detections, contours = detector.detect(frame)
        
        if frame_count > skiped_frame:
            tracker.update_tracks(detections, frame)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(
            frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
        )
        cv2.imshow("Frame", frame)

        end_time_first = time.time()
        print(end_time_first - start_time_first)
        keyboard = cv2.waitKey(1)
        if keyboard == 27:
            break
        elif keyboard == ord("s"):
            # cv2.imwrite("screenshots/threshold_frame_" + str(capture.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", thresh)
            cv2.imwrite("screenshots/original_frame_" + str(capture.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", frame)
        
    capture.release()
    cv2.destroyAllWindows()


def downsampling():
    file_name = "center_of_contour_thresh"
    image = cv2.imread("datasets/" + file_name + ".jpg")
    cv2.imshow("Original Image", image)

    scale = 8
    method = "resize"
    detector = Detector(1, 1, scale)
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
        cv2.imwrite("screenshots/" + file_name + "_x" + str(scale) + "_downsampled.jpg", downsampled_image)
        cv2.imwrite("screenshots/" + file_name + "_x" + str(scale) + "_original.jpg", image)
    else:    
        cv2.imwrite("screenshots/" + file_name + "_x" + str(scale) + "_resize_downsampled.jpg", downsampled_image)
        cv2.imwrite("screenshots/" + file_name + "_x" + str(scale) + "_resize_original.jpg", image)


    cv2.waitKey(0)    
    cv2.destroyAllWindows()


def contour_tracing():
    # i = 255  # white
    # o = 0  # black
    # image = np.array(
    #     [
    #         [o, o, o, o, o, o, o, o, o, o, o],
    #         [i, i, i, o, o, i, i, i, i, i, i],
    #         [i, o, i, o, o, i, o, o, o, o, i],
    #         [i, i, i, o, o, i, o, o, o, o, i],
    #         [o, o, o, o, o, i, o, o, o, o, i],
    #         [o, i, o, o, o, i, i, i, i, i, i],
    #         [o, o, o, o, o, o, o, o, o, o, o],
    #     ]
    # ).astype(np.uint8)

    image = cv2.imread("datasets/sample_01.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contour_tracing = ContourTracing()

    contours = contour_tracing.findCountourCustom(image)
    print(len(contours))

if __name__ == "__main__":
    downsampling()
