import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-if", "--ImageFolder", help="Image folder")
parser.add_argument("-rp", "--ResultPath", help="Result path")
args = parser.parse_args()

image_folder = args.ImageFolder
result_path = args.ResultPath

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(result_path, 0, 24, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
