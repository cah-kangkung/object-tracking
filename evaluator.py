import cv2 as cv2
import argparse
from sklearn.metrics import confusion_matrix
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", "--Name", help="File Name")
    parser.add_argument("-gtp", "--GTP", help="Groundtruth path")
    parser.add_argument("-pp", "--PP", help="Predicted path")
    args = parser.parse_args()

    ground_truth = cv2.imread(args.GTP, cv2.IMREAD_GRAYSCALE)
    predicted_image = cv2.imread(args.PP, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Ground Truth", ground_truth)
    cv2.imshow("Predicted Image", predicted_image)

    ret, thresh1 = cv2.threshold(ground_truth, 254, 1, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(predicted_image, 254, 1, cv2.THRESH_BINARY)

    ground_truth = thresh1.flatten()
    predicted_image = thresh2.flatten()

    tn, fp, fn, tp = confusion_matrix(ground_truth, predicted_image).ravel()

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    recall = (tp) / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * ((precision * recall) / (precision + recall))

    print(args.Name)
    print("accuracy", accuracy)
    print("recall", recall)
    print("precision", precision)
    print("f1", f1, "\n")

    with open("screenshots/morph_perfomance.txt", "a") as f:
        f.write(
            f"""# {args.Name} #\naccuracy  : {accuracy * 100}\nrecall    : {recall * 100}\nprecision : {precision * 100}\nf1 score  : {f1 * 100}\n\n"""
        )

    keyboard = cv2.waitKey(10)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
