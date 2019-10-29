import os 
import numpy as np
import matplotlib.pyplot
import pandas as pd

class Adaptive_Hopf_Oscillator():
    def __init__(self,
                 F,
                 phi = np.random.random(),
                 n = 1000000,
                 dt = 0.01,
                 mu = 1,
                 eps = 0.9,
                 omega_0 = 40):
        self.n = n
        self.dt = dt
        self.mu = mu
        self.eps = eps
        self.omega = omega_0
        self.z = self.z(omega_0)
        self.Aomega = np.zeros((1,self.n))
        self.Az = np.zeros((1, self.n))
        self.Aphi = np.zeros((1, self.n))
        self.t = np.arange(0,n)*self.dt
        self.F = F


    def z(self, omega):
        return np.exp(2j*np.pi*omega

    def dz(self):
        return True

    def adaptive_hopf_oscillator(self):

        return True
       
        
