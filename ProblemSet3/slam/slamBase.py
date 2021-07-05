"""
Sudhanva Sreesha
ssreesha@umich.edu
24-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018
"""

import numpy as np

from abc import ABC, abstractmethod


class SlamBase(ABC):
    def __init__(self, initial_state, alphas, beta):
        """
        :param slam_type: Which SLAM algorithm to run: ONLINE SLAM (ekf) or smoothing the full trajcetoryby using Factor graphs (sam).
        :param data_association: The type of data association to perform during the update step.
                                 Valid string choices include: {'known', 'nn', 'nndg', 'jcbb'}.
        :param update_type: The type of update to perform in the SLAM algorithm.
                            Valid string choices include: {'batch', 'sequential'}.
        :param Q: The observation noise covariance matrix: numpy.ndarray of size 2x2 for range and bearing measurements.
        """

        #assert isinstance(slam_type, str)
        #assert isinstance(data_association, str)
        #assert isinstance(update_type, str)
        #assert isinstance(Q, np.ndarray)

        #assert slam_type in {'ekf', 'sam'}
        #assert update_type in {'batch', 'sequential'}
        #assert data_association in {'known', 'ml', 'jcbb'}

        #self.slam_type = slam_type
        #self.da_type = data_association
        #self.update_type = update_type

        self.t = 0
        beta = np.array(beta)
        beta[1] = np.deg2rad(beta[1])
        self._Q = np.diag([beta[0], beta[1]])
        self._state = initial_state.mu.reshape(3)   #state vector (Vector of state poses and poses of landmarks)
        self._state_last = initial_state.mu         #pose of the robot in the previous step with adding delta (x[i-1]0 + delta)
        self._state_current = initial_state.mu      #pose of the robot calculated with state_last and u using get_prediction
        self._UX = initial_state.Sigma              #part of matrix A relalted to U and X
        self._ZX = np.array([])                     #part of matrix A relalted to Z and X
        self._ZM = []                               #part of matrix A relalted to Z and M
        self._alphas = alphas
        self._steps = 0                             #number of steps
        self._numObs = 0                            #number of observed observations in the last step
        self._LMs = []                              #list of observed landmarks
        self._LMs2 = []                             #"list of observed landmarks poses"
        self._X = initial_state.mu.reshape(3)       #part of vector state only with state poses of the robot
        self._M = []                                #part of vector state only with poses of observed landmarks
        self._prev_estim = initial_state.mu         #pose of the robot in the previous step before adding delta (x[i-1]0)
        self.b_a = np.zeros((3,1))                  #part of vector b only with elements A
        self.b_c = np.array([])                     #part of vector b only with elements C
        self.b = []                                 #vector b

        self.state_dim = 3  # The number of state variables: x, y, theta (initially).
        self.obs_dim = 2  # The number of variables per observation: range, bearing.
        self.lm_dim = 2  # The number of variables per landmark: x, y.

    # @abstractmethod
    # def predict(self, u, dt=None):
    #     """
    #     Updates mu_bar and Sigma_bar after taking a single prediction step after incorporating the control.
    #
    #     :param u: The control for prediction (format: np.ndarray([drot1, dtran, drot2])).
    #     :param dt: The time difference between the previous state and the current state being predicted.
    #     """
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def update(self, z):
    #     """
    #     Performs data association to figure out previously seen landmarks vs. new landmarks
    #     in the observations list and updates mu and Sigma after incorporating them.
    #
    #     :param z: Observation measurements (format: numpy.ndarray of size Kx3
    #               observations where each row is [range, bearing, landmark_id]).
    #     """
    #
    #     raise NotImplementedError()

    @abstractmethod
    def step(self, u, z):
        """
        Performs data association to figure out previously seen landmarks vs. new landmarks
        in the observations list and updates mu and Sigma after incorporating them.

        :param z: Observation measurements (format: numpy.ndarray of size Kx3
                  observations where each row is [range, bearing, landmark_id]).
        """

        raise NotImplementedError()
