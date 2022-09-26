from pprint import pprint
from kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2 as cv2


class Track(object):
    def __init__(self, prediction, track_id):
        self.track_id = track_id
        self.KF = KalmanFilter(1)
        self.prediction = np.asarray(prediction)  # in case of first prediction, takes the the first detection


class Tracker(object):
    def __init__(self):
        self.tracks = []
        self.track_id = 0

    def update_tracks(self, detections, frame):
        # If there are no tracks, check for detection and create track object
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                self.track_id += 1
                track = Track(detections[i], self.track_id)
                self.tracks.append(track)

        # Crate cost matrix using distance between predicted and detected object centroids
        n_length = len(self.tracks)
        m_length = len(detections)
        cost = np.zeros(shape=(n_length, m_length))  # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                difference = self.tracks[i].prediction[:2] - detections[j][:2]
                distance = np.sqrt(difference[0][0] * difference[0][0] + difference[1][0] * difference[1][0])
                cost[i][j] = distance
        cost = (0.5) * cost

        # assign using Hungarian Algorithm
        # https://www.youtube.com/watch?v=cQ5MsiGaDY8
        assignment = []
        for _ in range(n_length):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if assignment[i] != -1:
                self.tracks[i].prediction = self.tracks[i].KF.update(detections[assignment[i]])
            else:
                self.tracks[i].prediction = self.tracks[i].KF.update(np.array([[0], [0], [0], [0]]))
            print()

        for track in self.tracks:
            trackPrediction = np.array(track.prediction)
            x = int(trackPrediction[0][0])
            y = int(trackPrediction[1][0])
            w = int(trackPrediction[2][0])
            h = int(trackPrediction[3][0])
            cv2.rectangle(
                frame,
                (x - int((w / 2)), y - int((h / 2))),
                ((x - int((w / 2))) + w, (y - int((h / 2))) + h),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                str(track.track_id),
                (x - int((w / 2)), y - int((h / 2))),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
