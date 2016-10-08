import numpy as np
import defaultdict
import pandas as pd

class PLC:
    def __init__(self, nX, nY, nW, nR, nT, nC):
        self.X = np.zeros(nX, dtype="bool")
        self.Y = np.zeros(nY, dtype="bool")
        self.W = np.zeros(nD, dtype="uint16")
        self.R = np.zeros(nR, dtype="float32")
        self.T = np.zeros(nT, dtype="uint16")
        self.T_u = np.zeros(nC, dtype="bool")
        self.T_a = np.zeros(nT, dtype="bool")
        self.C = np.zeros(nC, dtype="uint16")
        self.C_u = np.zeros(nC, dtype="bool")
        self.C_a = np.zeros(nC, dtype="bool")
        self.Archive = defaultdict(list)
    def __len__(self):
        return len(self.Archive['timestamp'])
    def nextT(self, timestamp):
        self.Archive['X'].append(self.X.copy())
        self.Archive['Y'].append(self.Y.copy())
        self.Archive['W'].append(self.W.copy())
        self.Archive['R'].append(self.R.copy())
        self.Archive['T'].append(self.T.copy())
        self.Archive['T_u'].append(self.T_u.copy())
        self.Archive['T_a'].append(self.T_a.copy())
        self.Archive['C'].append(self.C.copy())
        self.Archive['C_u'].append(self.T_a.copy())
        self.Archive['C_a'].append(self.C_a.copy())
        self.Archive['timestamp'].append(timestamp)
