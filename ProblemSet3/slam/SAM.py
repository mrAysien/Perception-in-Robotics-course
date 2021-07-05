"""
This file implements the Extended Kalman Filter.
"""

import numpy as np
from scipy import linalg

from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import get_LM_pose
from tools.task import wrap_angle
from field_map import FieldMap
from tools.objects import Gaussian


def get_Jacob_H(state, lm_id):
    """
        Calculate the jacobian of observation model function with respect to the state variables.
    """
    lm_id = int(lm_id)
    field_map = FieldMap(4)
    dx = field_map.landmarks_poses_x[lm_id] - state[0]
    dy = field_map.landmarks_poses_y[lm_id] - state[1]

    A = (dx) ** 2 + (dy) ** 2

    return np.array([[-dx / np.sqrt(A), -dy / np.sqrt(A), 0],
                     [dy / A, -dx / A, -1]])

def get_Jacob_J(state, lm_id):
    """
            Calculate the jacobian of observation model function with respect to the landmark.
    """
    lm_id = int(lm_id)
    field_map = FieldMap(4)
    dx = field_map.landmarks_poses_x[lm_id] - state[0]
    dy = field_map.landmarks_poses_y[lm_id] - state[1]

    A = (dx) ** 2 + (dy) ** 2

    return np.array([[dx / np.sqrt(A), dy / np.sqrt(A)],
                     [-dy / A, dx / A]]).reshape(2,2)

def get_Jacob_Mot(state, u):
    x, y, theta = state
    drot1, dtran, drot2 = u
    if dtran == 0:
        dtran = 1e-4
    theta +=drot1
    return np.array([[-dtran*np.sin(theta), np.cos(theta), 0],
                     [ dtran*np.cos(theta), np.sin(theta), 0],
                     [ 1, 0, 1]])

def get_Jacob_G(state, u):
    """
            Calculate the jacobian of transformation function with respect to the state variables.
    """

    return np.array([[1, 0, -u[1] * np.sin(float(state[2] + u[0]))],
                     [0, 1, u[1] * np.cos(float(state[2] + u[0]))],
                     [0, 0, 1]])
def InvTrans(Cov):
    """
                Calculate the the matrix in power -T/2
    """
    Cov = np.linalg.inv(Cov)
    L = linalg.cholesky(Cov,lower=True)
    return L.T

def Calc_b(self, z, u):
    """
                Calculate the vector b
    """

    M = get_motion_noise_covariance(u, self._alphas)
    M = InvTrans(M)
    N = z[:, 0].size
    Q = InvTrans(self._Q[:2, :2])
    a = self._state_current - get_prediction(self._prev_estim.reshape(3), u)
    self.b_a = np.vstack((self.b_a, np.dot(M, a.reshape(3, 1))))
    if self._steps == 0:
        for i in range(N):
            if i == 0:
                c = z[i, :2].reshape(2, 1) - get_expected_observation(self._state_current, FieldMap(4), z[i, 2])[
                                             :2].reshape(2, 1)
                self.b_c = np.dot(Q, c)
            else:
                c = z[i, :2].reshape(2, 1) - get_expected_observation(self._state_current, FieldMap(4), z[i, 2])[
                                             :2].reshape(2, 1)
                self.b_c = np.vstack((self.b_c, np.dot(Q, c)))
    else:
        for i in range(N):
            c = z[i, :2].reshape(2, 1) - get_expected_observation(self._state_current, FieldMap(4), z[i, 2])[
                                         :2].reshape(2, 1)
            self.b_c = np.vstack((self.b_c, np.dot(Q, c)))

    self.b = np.vstack((self.b_a, self.b_c))

def Calc_delta(self, A):
    """
                    Recalculate state vector (Vector of state poses and poses of landmarks)
    """

    delta = np.linalg.lstsq(A, self.b, rcond=None)[0]
    self._state +=delta.reshape(delta.size,)

def Calc_A(self, u, z):
    """
                        Calculate the matrix A
    """

    M = get_motion_noise_covariance(u, self._alphas)
    M = InvTrans(M)
    G = get_Jacob_G(self._prev_estim, u)
    self._UX = np.hstack((self._UX, np.zeros((3 * (self._steps + 1), 3))))
    self._UX = np.vstack((self._UX, np.zeros((3, 3 * (self._steps + 2)))))
    self._UX[(3 * (self._steps + 1)):, (3 * (self._steps + 1)):] = np.dot(M,-np.eye(3))
    self._UX[(3 * (self._steps + 1)):, (3 * self._steps):(3 * (self._steps + 1))] = np.dot(M,G)

    N = z[:, 0].size
    Q = InvTrans(self._Q[:2, :2])

    if self._steps == 0:
        self._ZX = np.zeros((2 * N, 3 * N))
        self._ZM = np.zeros((2 * N, 2 * N))
        for i in range(N):
            H = get_Jacob_H(self._state_current, z[i, 2])
            self._ZX[2 * i:2 * (i + 1), 3:] = np.dot(Q,H)
            J = get_Jacob_J(self._state_current, z[i, 2])
            self._LMs2 = np.append(self._LMs2, [2 * int(z[i, 2]), 2 * int(z[i, 2]) + 1])
            self._LMs = np.append(self._LMs, [int(z[i, 2])])
            self._ZM[2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] = np.dot(Q,J)
            self._M = np.append(self._M, get_LM_pose(self._state_current, z[i, :]))
        i = np.argsort(self._LMs2)
        self._LMs2 = self._LMs2[i]
        self._LMs = self._LMs[np.argsort(self._LMs)]
        self._ZM = self._ZM[:, i]
        self._M = self._M[i]
        # print(M)
        # print(Q)

    else:
        self._ZX = np.hstack((self._ZX, np.zeros((2 * (self._numObs), 3))))
        self._ZX = np.vstack((self._ZX, np.zeros((2 * N, 3 * (self._steps + 2)))))
        for i in range(N):
            H = get_Jacob_H(self._state_current, z[i, 2])
            self._ZX[(2 * (self._numObs + i)):(2 * (self._numObs + i + 1)), 3 * (self._steps + 1):] = np.dot(Q,H)

        if len(self._LMs) == 8:
            for j in range(N):
                J = get_Jacob_J(self._state_current, z[j, 2])
                i = 0
                while i < len(self._LMs):
                    if self._LMs[i] == z[j, 2]:
                        self._ZM = np.vstack((self._ZM, np.zeros((2, 2 * len(self._LMs)))))
                        self._ZM[2 * (self._numObs + j):, 2 * i:2 * (i + 1)] = np.dot(Q,J)
                        break
                    i += 1
        else:
            for j in range(N):
                J = get_Jacob_J(self._state_current, z[j, 2])
                i = 0
                while i < len(self._LMs):
                    if self._LMs[i] == z[j, 2]:
                        self._ZM = np.vstack((self._ZM, np.zeros((2, 2 * len(self._LMs)))))
                        self._ZM[2 * (self._numObs + j):, 2 * i:2 * (i + 1)] = np.dot(Q,J)
                        break
                    i += 1
                else:
                    self._LMs = np.append(self._LMs, [int(z[j, 2])])
                    self._LMs2 = np.append(self._LMs2, [2 * int(z[j, 2]), 2 * int(z[j, 2]) + 1])
                    self._ZM = np.hstack((self._ZM, np.zeros((2 * (self._numObs + j), 2))))
                    self._ZM = np.vstack((self._ZM, np.zeros((2, 2 * len(self._LMs)))))
                    self._ZM[2 * (self._numObs + j):, 2 * (len(self._LMs) - 1):] = np.dot(Q,J)

                    self._M = np.append(self._M, get_LM_pose(self._state_current, z[j, :]))

                i = np.argsort(self._LMs2)
                self._LMs2 = self._LMs2[i]
                self._LMs = self._LMs[np.argsort(self._LMs)]
                self._ZM = self._ZM[:, i]

                self._M = self._M[i]


    A = self._UX
    A = np.hstack((A, np.zeros((3 * (self._steps + 2), len(self._LMs2)))))
    A = np.vstack((A, np.hstack((self._ZX, self._ZM))))
    return A
class SAM(SlamBase):
    def step(self, u, z):
        self._state_current = get_prediction(self._state_last.reshape(3), u.reshape(3))
        A = Calc_A(self, u, z)
        self._X = np.append(self._X, self._state_current)

        self._state = np.append(self._X, self._M)
        Calc_b(self, z, u)
        Calc_delta(self, A)
        self._X, self._M = self._state[:self._X.size], self._state[self._X.size:]
        self._prev_estim = self._state_current
        self._state_last = self._state[3*(self._steps+1):3*(self._steps+2)]
        # if self._steps == 0:
        #     print(A)
        self._steps += 1
        self._numObs += z[:, 0].size
