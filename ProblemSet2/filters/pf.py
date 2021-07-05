"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle

def CreateIndex(W, A):
    ind = []
    C = np.append([], np.cumsum(W))
    k = 0
    r = (np.random.rand() + np.arange(A)) / A
    for i in r:
        while k < len(C) and i > C[k]:
            k += 1
        ind += [k - 1]
    return ind

class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)

        # TODO add here specific class variables for the PF
        self.particles = np.ones((num_particles, 3)) * initial_state.mu[:, 0]

        self.num_particles = num_particles

        self.weights_m = np.array([1/num_particles for i in range(self.num_particles)])


    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        for i in range(self.num_particles):
            self.particles[i] = sample_from_odometry( self.particles[i], u, self._alphas)

        state_norm = get_gaussian_statistics(self.particles)

        self._state_bar.mu = state_norm.mu
        self._state_bar.Sigma = state_norm.Sigma

    def update(self, z):
        # TODO implement correction step
        z_predict = np.zeros(self.num_particles)
        y = []
        for i in range (self.num_particles):
            z_predict[i], _ = get_observation(self.particles[i], z[1])
            y.append(wrap_angle(z_predict[i]-z[0]))
        weights = gaussian().pdf(y)

        weigths = np.array(weights) / np.sum(weights)

        index = CreateIndex(weigths, self.num_particles)

        self.particles = self.particles[index]
        state_norm = get_gaussian_statistics(self.particles)
        self._state.mu = state_norm.mu
        self._state.Sigma = state_norm.Sigma