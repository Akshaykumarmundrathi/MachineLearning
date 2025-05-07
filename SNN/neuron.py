import numpy as np

class neuron:
    def __init__(self):
        self.Pth = 6           # Lower threshold for 5x5 patterns
        self.t_ref = 2
        self.t_rest = -1
        self.P = 0.0
        self.D = 0.5
        self.Pmin = -1
        self.Prest = 0.0

    def check(self):
        if self.P >= self.Pth:
            self.P = self.Prest
            return 1
        elif self.P < self.Pmin:
            self.P = self.Prest
            return 0
        else:
            return 0

    def inhibit(self):
        self.P = self.Pmin

    def initial(self):
        self.t_rest = -1
        self.P = self.Prest
