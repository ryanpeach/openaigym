import numpy as np

def dict_diff(d1, d2):
    out = {}
    for k, v in d2.iteritems():
        out[k] = v - d1[k]
    return out

class Data_Recorder:
    def __init__(self, control, monitor, usr_control = [], usr_monitor = [], plc_control = []):
        self.usr_control, self.usr_monitor, self.plc_control = usr_control, usr_monitor, plc_control
        # TODO: Assert no item in plc_control is in control or usr_control
        self.control = self.control.keys()
        self.monitor = self.monitor.keys()
        self.both = list(set(control).union(monitor))
        state = control.copy()
        state.update(monitor)
        self.data = [state]
        
    def __setitem__(self, key, val):
        """ You can only set keys in self.control """
        if key in self.control:
            self.data[-1][key] = float(val)
        else:
            raise KeyError
        
    def __getitem__(self, key):
        """ You can only read keys in self.monitor """
        if key in self.monitor:
            return self.data[-1][key]
        else:
            raise KeyError
    
    def setState(self, state):
        """ Calls setitem over a dictionary state """
        for k, v in state.iteritems():
            self[k] = v
            
    def nextT(self):
        """ Saves last state and creates identical new state """
        self.data.append(self.data[-1].copy())
        
    def getControlT(self, t = 0):
        out = []
        for k in self.control:
            out.append(self.data[-1][k])
        return np.array(out, dtype="float32"), self.control
        
    def getMonitorT(self, t = 0):
        out = []
        for k in self.monitor:
            out.append(self.data[-1][k])
        return np.array(out, dtype="float32"), self.monitor
    
    def getT(self, t = 0):
        out = []
        for k in self.monitor:
            out.append(self.data[-1][k])
        return np.array(out, dtype="float32"), self.both
        
    def __len__(self):
        return len(self.data)
        
    def __repr__(self):
        return self.data[-1].copy()