from kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np


class Track(object):
    def __init__(self, prediction, track_id):
        self.track_id = track_id
        self.KF = KalmanFilter(1, 0, 0)
        self.state = np.asarray(prediction)  # in case of first prediction, takes the the first detection


class Tracker(object):
    def __init__(self):
        self.tracks = []
        self.track_id = 0

    def update_tracks(self, detections):
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
                difference = self.tracks[i].state[:2] - detections[j][:2]
                distance = np.sqrt(difference[0][0] * difference[0][0] + difference[1][0] * difference[1][0])
                cost[i][j] = distance
        cost = (0.5) * cost

        assignment = []
        for _ in range(n_length):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if assignment[i] != -1:
                self.tracks[i].state = self.tracks[i].KF.update(detections[assignment[i]])
            else:
                self.tracks[i].state = self.tracks[i].KF.update(np.array([[0], [0], [0], [0]]))
