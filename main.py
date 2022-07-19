from matplotlib.pyplot import contour
import numpy as np
import cv2 as cv2
import time
import copy as copy

from contour_tracing import ContourTracing


def main():

    original_image = cv2.imread("datasets/DeepFish/Segmentation/images/valid/7623_F2_f000320.jpg")
    image = cv2.imread("datasets/DeepFish/Segmentation/masks/valid/7623_F2_f000320.png")
    # image = cv2.imread("datasets/Ga5Pe.png")
    resize_value = 2
    height, width, layers = image.shape
    new_h = height / resize_value
    new_w = width / resize_value
    image = cv2.resize(image, (int(new_w), int(new_h)))
    original_image = cv2.resize(original_image, (int(new_w), int(new_h)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # i = 255  # black
    # o = 0  # black
    # image = np.array(
    #     [
    #         [o, o, o, o, o, o, o, o, o, o, o],
    #         [o, o, o, i, i, i, i, i, o, o, o],
    #         [o, o, i, i, i, i, i, i, i, o, o],
    #         [o, i, i, i, i, i, i, i, i, i, o],
    #         [o, i, i, i, o, o, o, i, i, i, o],
    #         [o, i, i, i, o, o, o, i, i, i, o],
    #         [o, i, i, i, o, o, o, i, i, i, o],
    #         [o, i, i, i, i, i, i, i, i, i, o],
    #         [o, o, i, i, i, i, i, i, i, o, o],
    #         [o, o, o, i, i, i, i, i, o, o, o],
    #         [o, o, o, o, o, o, o, o, o, o, o],
    #     ]
    # ).astype(np.uint8)

    start_time = time.time()
    contour_tracing = ContourTracing()
    contours = contour_tracing.findCountourCustom(image)

    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 255), 1)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time_alternate = time.time()
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("--- %s seconds ---" % (time.time() - start_time_alternate))

    h, w = image.shape
    cv2.imshow(f"Original Image ({w}, {h})", original_image)
    cv2.imshow(f"Mask ({w}, {h})", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
