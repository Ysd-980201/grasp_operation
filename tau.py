#!/usr/bin/env python

import numpy as np


class NoTC:
    def update(self, dmp, dt):
        return 0


class TCVelAccConstrained:

    def __init__(self, nominal_tau, gamma_nominal, gamma_a, v_max, a_max, eps=0.001):
        self.nominal_tau = nominal_tau
        self.gamma_nominal = gamma_nominal
        self.gamma_a = gamma_a
        self.eps = eps
        self.v_max = v_max.reshape((len(v_max), 1))
        # print(self.v_max)
        self.a_max = a_max.reshape((len(a_max), 1))

    def generate_matrices(self, dmp, nextstate, dt):#用来计算x，v，和tau* dv
        
        A = np.vstack((-dmp.vvv(), dmp.vvv()))
        B = np.vstack((-self.a_max, -self.a_max))
        C = np.vstack((dmp.hhh(), -dmp.hhh()))
        D = np.vstack((-self.v_max, -self.v_max))
        A_next = np.vstack((-nextstate[0], nextstate[0]))
        C_next = np.vstack((nextstate[1], -nextstate[1]))
        return A, B, C, D, A_next, C_next    
    
    def tau_dot(self, dmp, nextstate, dt):

        A, B, C, D, A_next, C_next = self.generate_matrices(dmp, nextstate, dt)

        # Acceleration bounds
        i = np.squeeze(A < 0)
        if i.any():
            taud_min_a = np.max(- (B[i] * dmp.tau ** 2 + C[i]) / A[i])
        else:
            taud_min_a = -np.inf
        i = np.squeeze(A > 0)
        if i.any():
            taud_max_a = np.min(- (B[i] * dmp.tau ** 2 + C[i]) / A[i])
        else:
            taud_max_a = np.inf
        # Velocity bounds
        i = range(len(A_next))
        tau_min_v = np.max(-A_next[i] / D[i])
        taud_min_v = (tau_min_v - dmp.tau) / dt
        # Feasibility bounds
        ii = np.arange(len(A_next))[np.squeeze(A_next < 0)]
        jj = np.arange(len(A_next))[np.squeeze(A_next > 0)]
        tau_min_f = -np.inf
        for i in ii:
            for j in jj:
                num = C_next[i] * abs(A_next[j]) + C_next[j] * abs(A_next[i])
                if num > 0:
                    den = abs(B[i] * A_next[j]) + abs(B[j] * A_next[i])
                    tmp = np.sqrt(num / den)
                    if tmp > tau_min_f:
                        tau_min_f = tmp
        taud_min_f = (tau_min_f - dmp.tau) / dt
        # Nominal bound
        taud_min_nominal = (dmp.nominal_tau - dmp.tau) / dt

        taud_min = np.max((taud_min_a, taud_min_v, taud_min_f.squeeze(), taud_min_nominal))

        # Base update law
        ydd_bar = dmp.hhh() / (dmp.tau**2 * self.a_max)
        if self.gamma_a > 0:
            pot_a = self.gamma_a * \
                np.sum(ydd_bar ** 2 * (np.exp(dmp.tau) - np.exp(-dmp.tau)) / ((np.exp(dmp.tau) + np.exp(-dmp.tau))*np.maximum(1 - ydd_bar ** 2, self.gamma_a * self.eps * np.ones((len(ydd_bar), 1)))))
        else:
            pot_a = 0
        taud = (self.gamma_nominal * (dmp.nominal_tau - dmp.tau) + self.gamma_a * pot_a) *(np.exp(dmp.tau) - np.exp(-dmp.tau)) / (np.exp(dmp.tau) + np.exp(-dmp.tau))
        # Saturate
        taud = np.min((taud, taud_max_a))
        taud = np.max((taud, taud_min))

        return taud
#(np.exp(dmp.tau) - np.exp(-dmp.tau)) / (np.exp(dmp.tau) + np.exp(-dmp.tau))
