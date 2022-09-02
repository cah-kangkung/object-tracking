import copy as copy
import cv2 as cv2
from detector import Detector
from contour_tracing import ContourTracing
from tracker import Tracker


def main():
    # Input
    video_path = "datasets/fish_tank_02.mp4"
    kernel_erosion_size = 0
    kernel_dilation_size = 7
    resize_value = 2

    capture = cv2.VideoCapture(video_path)
    detector = Detector(kernel_erosion_size, kernel_dilation_size)
    tracker = Tracker()

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        orig_frame = copy.copy(frame)

        height, width, layers = frame.shape
        new_h = height / resize_value
        new_w = width / resize_value
        frame = cv2.resize(frame, (int(new_w), int(new_h)))

        centers, contours, thresh = detector.detect(frame)

        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(
            frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
        )
        cv2.imshow("Frame", frame)

        # if len(centers) > 0:
        #     tracker.update_tracks(centers)

        keyboard = cv2.waitKey(50)
        if keyboard == 27:
            break
        elif keyboard == ord("s"):
            cv2.imwrite("screenshots/threshold_frame_" + str(capture.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", thresh)
            cv2.imwrite("screenshots/original_frame_" + str(capture.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", orig_frame)

    capture.release()
    cv2.destroyAllWindows()


def contour_tracing():
    original_image = cv2.imread("screenshots/original_frame_39.0.jpg")
    image = cv2.imread("screenshots/threshold_frame_39.0.jpg")

    # original_image = cv2.imread("datasets/DeepFish/Segmentation/images/valid/7623_F2_f000320.jpg")
    # image = cv2.imread("datasets/DeepFish/Segmentation/masks/valid/7623_F2_f000320.png")

    resize_value = 1
    height, width, layers = image.shape
    new_h = height / resize_value
    new_w = width / resize_value
    image = cv2.resize(image, (int(new_w), int(new_h)))
    original_image = cv2.resize(original_image, (int(new_w), int(new_h)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    contour_tracing = ContourTracing()
    contours = contour_tracing.findCountourCustom(image)

    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    h, w = image.shape
    cv2.imshow(f"Original Image ({w}, {h})", original_image)
    cv2.imshow(f"Mask ({w}, {h})", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    contour_tracing()
