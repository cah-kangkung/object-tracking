import copy as copy
import cv2 as cv2
from detector import Detector
from tracker import Tracker
from contour_tracing import ContourTracing
import numpy as np
import time
import os
import argparse
from sklearn.metrics import mean_squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--Mode", default=10, help="Wait key mode")
    parser.add_argument("-p", "--Path", help="Video path")
    parser.add_argument("-n", "--Name", help="Video name")
    parser.add_argument("-kes", "--KES", help="Kernel Erosion Size")
    parser.add_argument("-kds", "--KDS", help="Kernel Dilation Size")
    parser.add_argument("-s", "--Scale", help="Downsampling Scale")
    parser.add_argument("-ss", "--ScreenShots", help="Frame to be saved")
    args = parser.parse_args()

    # Input
    mode = int(args.Mode)
    name = args.Name
    video_path = args.Path
    kernel_erosion_size = int(args.KES)
    kernel_dilation_size = int(args.KDS)
    screenshots = []
    if args.ScreenShots:
        screenshots = args.ScreenShots.split(",")
    scale = int(args.Scale)
    downsampling_mode = "resize"
    max_frame_to_skip = 10

    capture = cv2.VideoCapture(video_path)
    detector = Detector(kernel_erosion_size, kernel_dilation_size, scale, downsampling_mode)
    tracker = Tracker(max_frame_to_skip)

    skiped_frame = 50
    frame_count = 0
    while True:
        ret, frame = capture.read()
        original_frame = copy.copy(frame)
        if frame is None:
            break
        frame_count += 1

        (
            detections,
            contours,
            fg_mask,
            contour_tracing_frame,
            eroded,
            dilated,
            downsampled_image,
            ct_elapse_time,
        ) = detector.detect(frame)

        if frame_count > skiped_frame:
            tracker.update_tracks(detections, frame, int(capture.get(cv2.CAP_PROP_POS_FRAMES)))

        current_frame = str(int(capture.get(cv2.CAP_PROP_POS_FRAMES)))

        # Frame
        cv2.rectangle(frame, (10, 2), (80, 20), (255, 255, 255), -1)
        cv2.putText(frame, current_frame, (13, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Count
        cv2.rectangle(frame, (100, 2), (130, 20), (255, 255, 255), -1)
        cv2.putText(
            frame,
            str(len(tracker.tracks)),
            (105, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )

        # Legend
        cv2.rectangle(frame, (150, 2), (440, 20), (255, 255, 255), -1)
        cv2.rectangle(frame, (155, 6), (175, 16), (255, 0, 0), -1)
        cv2.putText(frame, "=predicted", (180, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.rectangle(frame, (270, 6), (290, 16), (0, 0, 255), -1)
        cv2.putText(frame, "=KF", (295, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.rectangle(frame, (335, 6), (355, 16), (0, 255, 255), -1)
        cv2.putText(frame, "=detected", (355, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Scale
        cv2.rectangle(frame, (450, 2), (480, 20), (255, 255, 255), -1)
        cv2.putText(frame, f"x{str(scale)}", (455, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow("Frame", frame)
        cv2.imshow("GMM", fg_mask)
        cv2.imshow("Eroded", eroded)
        cv2.imshow("Dilated", dilated)

        keyboard = cv2.waitKey(mode)
        if keyboard == 27 or (int(current_frame) > int(screenshots[-1])):
            break
        elif keyboard == ord("s") or current_frame in screenshots:
            print("FRAME SS", current_frame)

            path_orig_gmm = f"screenshots/{name}/f{current_frame}"
            os.makedirs(path_orig_gmm, exist_ok=True)
            cv2.imwrite(
                f"{path_orig_gmm}/{name}_original_frame{current_frame}.jpg",
                original_frame,
            )
            cv2.imwrite(f"{path_orig_gmm}/{name}_gmm_frame{current_frame}.jpg", fg_mask)

            path_morph = f"screenshots/{name}/f{current_frame}/gmm_morph"
            os.makedirs(path_morph, exist_ok=True)
            cv2.imwrite(
                f"{path_morph}/{name}_eroded_{kernel_erosion_size}x{kernel_dilation_size}_frame{current_frame}.jpg",
                eroded,
            )
            cv2.imwrite(
                f"{path_morph}/{name}_dilated_{kernel_erosion_size}x{kernel_dilation_size}_frame{current_frame}.jpg",
                dilated,
            )

            path_downsample = f"screenshots/{name}/f{current_frame}/downsample"
            os.makedirs(path_downsample, exist_ok=True)
            cv2.imwrite(
                f"{path_downsample}/{name}_downsample_x{scale}_m{kernel_erosion_size}x{kernel_dilation_size}_frame{current_frame}.jpg",
                downsampled_image,
            )

            path_contour_tracing = f"screenshots/{name}/f{current_frame}/contour_tracing"
            os.makedirs(path_contour_tracing, exist_ok=True)
            with open(f"{path_contour_tracing}/performance.txt", "a") as f:
                f.write(
                    f"{name}_downsample_x{scale}_m{kernel_erosion_size}x{kernel_dilation_size}_frame{current_frame} = {ct_elapse_time} seconds & ({len(contours)}) contours \n"
                )
            cv2.imwrite(
                f"{path_contour_tracing}/{name}_contour_downsample_x{scale}_m{kernel_erosion_size}x{kernel_dilation_size}_frame{current_frame}.jpg",
                contour_tracing_frame,
            )

            path_result = f"screenshots/{name}/f{current_frame}/result_kf"
            os.makedirs(path_result, exist_ok=True)
            cv2.imwrite(
                f"{path_result}/{name}_result_kf_x{scale}_m{kernel_erosion_size}x{kernel_dilation_size}_frame{current_frame}.jpg",
                frame,
            )

    x_rmse = mean_squared_error(tracker.x_detected, tracker.x_predicted, squared=False)
    y_rmse = mean_squared_error(tracker.y_detected, tracker.y_predicted, squared=False)
    print(x_rmse, y_rmse)
    total_rmse = np.sqrt(x_rmse**2 + y_rmse**2)

    average_total_objects = np.sum(tracker.number_of_objects) / len(tracker.number_of_objects)

    os.makedirs(f'screenshots/{name}', exist_ok=True)
    with open(f'screenshots/{name}/kf_rmse.txt', "a") as f:
        f.write(
            f"{name}_downsample_x{scale}_m{kernel_erosion_size}x{kernel_dilation_size}_frame{current_frame} = total {x_rmse} + {y_rmse} > {total_rmse} | average {average_total_objects} \n"
        )

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
