import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter(object):
    def __init__(self, dt, accel_x, accel_y):
        super(KalmanFilter, self).__init__()

        self.dt = dt

        # State with initial value
        self.x = np.matrix(
            [[0], [0], [0], [0], [0], [0]]  # x position  # y position  # half w  # half h  # x velocity  # y velocity
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

        # Control vector, constant acceleration for x and y
        self.u = np.matrix([[accel_x], [accel_y]])

        # Control transition matrix
        self.B = np.matrix([[self.dt ** 2 / 2, 0], [0, self.dt ** 2 / 2], [0, 0], [0, 0], [self.dt, 0], [0, self.dt],])

        # The error covariance matrix that is Identity for initial value.
        self.P = np.eye(self.F.shape[1])

        # Measurement transition matrix
        self.H = np.matrix([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],])

        self.Q = np.eye(self.F.shape[1])
        self.R = np.eye(4)

    def predict(self):
        # Predict state
        self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)

        # Update state cov matrix
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        # Residual
        y = z - np.dot(self.H, self.x)

        # Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R))

        # Update state
        self.x = np.round(self.x + np.dot(K, y))

        I = np.eye(self.H.shape[1])

        # Update state cov matrix
        self.P = I - (K * self.H) * self.P

        return self.x

