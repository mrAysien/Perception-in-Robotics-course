"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle
from field_map import FieldMap
from tools.objects import Gaussian

def get_Jacob(state, lm_id):
    
    lm_id = int(lm_id)
    field_map = FieldMap()
    dx = field_map.landmarks_poses_x[lm_id] - state[0]
    dy = field_map.landmarks_poses_y[lm_id] - state[1]
    
    A = ((dx)**2 + (dy)**2)
    
    return np.array([[dy/A, -dx/A, -1]])

class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        

        
        G = np.array([[1, 0, -u[1]*np.sin(float(self._state.mu[2] + u[0]))],
                      [0, 1,  u[1]*np.cos(float(self._state.mu[2] + u[0]))],
                      [0, 0,  1]])
        R = get_motion_noise_covariance(u, self._alphas)
        V = np.array([[-u[1]*np.sin(float(self._state.mu[2] + u[0])), np.cos(float(self._state.mu[2] + u[0])), 0],
                      [u[1]*np.cos(float(self._state.mu[2] + u[0])),  np.sin(float(self._state.mu[2] + u[0])), 0],
                      [1, 0,  1]])
        R = np.dot(V, np.dot(R, V.T))
        self._state_bar.mu = get_prediction(self.mu, u) #.reshape(3,1)
        self._state_bar.Sigma = np.dot(G, np.dot(self._state.Sigma, G.T)) + R


    def update(self, z):
        # TODO implement correction step

        H = get_Jacob(self._state_bar.mu, z[1])
        K = np.dot(self._state_bar.Sigma, H.T)*np.linalg.inv(np.dot(H, np.dot(self._state_bar.Sigma, H.T)) + self._Q)
        self._state.mu = (self._state_bar.mu + K.reshape(3).dot(z[0] - get_expected_observation(self._state_bar.mu, z[1])[0])).reshape(3,1)
        self._state.Sigma = np.dot((np.eye(3) - np.dot(K.reshape(3,1),H.reshape(1,3))),self._state_bar.Sigma)
