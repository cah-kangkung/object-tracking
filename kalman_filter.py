import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter(object):
    def __init__(self, dt):

        self.dt = dt

        # State with initial value
        self.x = np.matrix(
            [[0], [0], [0], [0], [1], [1]]  # x position  # y position  # w  # h  # x velocity  # y velocity
        )

        # State transition matrix
        self.F = np.matrix(
            [
                [1, 0, 0, 0, dt, 0],
                [0, 1, 0, 0, 0, dt],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # Unused
        # Control vector, constant acceleration for x and y
        # self.u = np.matrix([[accel_x], [accel_y]])

        # Unused
        # Control transition matrix
        # self.B = np.matrix([[self.dt ** 2 / 2, 0], [0, self.dt ** 2 / 2], [0, 0], [0, 0], [self.dt, 0], [0, self.dt],])

        # Measurement matrix consist of (x, y) coordinate, width, and height of the detected object
        self.z = np.matrix([[0], [0], [0], [0]])

        # Measurement transition matrix
        self.H = np.eye(self.z.shape[0], self.F.shape[1])

        # The error covariance matrix that is Identity for initial value.
        self.P = np.eye(self.F.shape[1])

        self.Q = np.eye(self.F.shape[1])
        self.R = np.eye(self.z.shape[0])

    def predict(self):
        # Predict state
        # self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        # print("old state\n", self.x)
        self.x = np.dot(self.F, self.x)
        # print("Predicted state\n", self.x)

        # Predict state cov matrix
        # print("old cov matrix\n", self.P)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        # print("Predicted cov matrix\n", self.P)
        return self.x

    def update(self, z):
        # Residual
        y = z - np.dot(self.H, self.x)

        # Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R))

        # Update state
        self.x = np.round(self.x + np.dot(K, y))
        # print("Updated state\n", self.x)

        I = np.eye(self.H.shape[1])

        # Update state cov matrix
        self.P = (I - K * self.H) * self.P
        # print("Updated cov matrix\n", self.P)

        return self.x

