import cv2 as cv2
from bs4 import BeautifulSoup
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--Path", help="XML path")
    parser.add_argument("-cfip", "--CFIP", help="Current Frame Image Path")
    parser.add_argument("-n", "--Name", help="Video Name")
    parser.add_argument("-f", "--Frame", help="Frame to be parse")
    args = parser.parse_args()

    current_frame_image = cv2.imread(args.CFIP)
    cv2.imshow("original", current_frame_image)

    with open(args.Path, "r") as f:
        data = f.read()

    bs_data = BeautifulSoup(data, "xml")

    contours = []
    frame_tag = bs_data.find("frame", {"id": args.Frame})
    objects = frame_tag.find_all("object")
    for object in objects:
        tag_value = object.find("contour").string
        coordinates = tag_value.split(",")

        new_coordinates = []
        for coordinate in coordinates:
            coordinate = coordinate.split(" ")
            new_coordinates.append(np.array([int(coordinate[0]), int(coordinate[1])]))

        contours.append(np.array(new_coordinates))

    contours = np.array(contours)
    row, col, _ = current_frame_image.shape
    binary_image = np.zeros((row, col))
    cv2.drawContours(
        current_frame_image, contours, -1, (0, 255, 255), thickness=cv2.FILLED
    )
    cv2.drawContours(binary_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    cv2.imshow("original", current_frame_image)
    cv2.imshow("xmlParser", binary_image)
    keyboard = cv2.waitKey(0)
    if keyboard == ord("s"):
        cv2.imwrite(f"groundtruth/{args.Name}_{args.Frame}.jpg", binary_image)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
