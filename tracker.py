from kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2 as cv2


class Track(object):
    def __init__(self, prediction, track_id):
        self.track_id = track_id
        self.KF = KalmanFilter(1)
        self.prediction = np.asarray(prediction)  # in case of first prediction, takes the the first detection
        self.frames_skipped = 0


class Tracker(object):
    def __init__(self, max_frames_to_skip):
        self.tracks = []
        self.max_frames_to_skip = max_frames_to_skip
        self.track_id = 0

    def update_tracks(self, detections, frame):
        # If there are no tracks, check for detection and create track object
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                self.track_id += 1
                track = Track(detections[i], self.track_id)
                self.tracks.append(track)

        self.show_boundingbox(frame, self.tracks, (255, 0, 0)) 

        # Crate cost matrix using distance and area between predicted and detected objects
        n_length = len(self.tracks)
        m_length = len(detections)
        cost = np.zeros(shape=(n_length, m_length))  # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                difference = self.tracks[i].prediction[:2] - detections[j][:2]
                distance = np.sqrt(difference[0][0] * difference[0][0] + difference[1][0] * difference[1][0])
                
                lh_prediction = self.tracks[i].prediction[2:4]
                lh_detection = detections[j][2:4]
                area = np.absolute((lh_prediction[0][0] * lh_prediction[1][0]) - (lh_detection[0][0] * lh_detection[1][0]))
                
                cost[i][j] = (0.8 * distance) + (0.2 * area)

        # Assign using Hungarian Algorithm
        # https://www.youtube.com/watch?v=cQ5MsiGaDY8
        assignment = []
        for _ in range(n_length):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment
        for i in range(len(assignment)):
            if (assignment[i] == -1):
                self.tracks[i].frames_skipped += 1

         # If tracks are not detected for long time, remove them
        deleted_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].frames_skipped > self.max_frames_to_skip):
                deleted_tracks.append(i)
        if len(deleted_tracks) > 0: 
            for index in deleted_tracks:
                if index < len(self.tracks):
                    del self.tracks[index]
                    del assignment[index]
                else:
                    print("ERROR: index is greater than length of tracks")

        # Look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.track_id)
                self.track_id += 1
                self.tracks.append(track)

        # Update KF
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if assignment[i] != -1:
                # if a prediction assigned to a detection, update prediction with that detection
                self.tracks[i].prediction = self.tracks[i].KF.update(detections[assignment[i]])
            else:
                # else just update with no detection
                self.tracks[i].prediction = self.tracks[i].KF.x

        self.show_boundingbox(frame, self.tracks, (0, 0, 255))

    def show_boundingbox(self, frame, tracks, color):
        for track in tracks:
            trackPrediction = np.array(track.prediction)
            x = int(trackPrediction[0][0])
            y = int(trackPrediction[1][0])
            w = int(trackPrediction[2][0])
            h = int(trackPrediction[3][0])
            cv2.rectangle(
                frame,
                (x - int((w / 2)), y - int((h / 2))),
                ((x - int((w / 2))) + w, (y - int((h / 2))) + h),
                color,
                2,
            )
            cv2.putText(
                frame,
                str(track.track_id),
                (x - int((w / 2)), y - int((h / 2))),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )