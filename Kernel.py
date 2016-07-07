import numpy as np
import math


class Kernel:
    position = np.array  # [x,y]
    color = np.array  # [r,g,b]

    def __init__(self, position, color):
        self.position = position
        self.color = color

    def move(self, position):
        self.position = position

    def set_color(self, color):
        self.color = color

    def calculate_kernel(self, hs, hr):
        '''

        :param hs: constant de entrada para x,y  d=2
        :param hr: input constant for r,g,b d=3
        :return: K hs,hr
        '''
        c = 0
        const = c / (math.pow(hs, 2) * math.pow(hr, 3))
        d2_const = np.linalg.norm(self.position / hs, 2)
        d3_const = np.linalg.norm(self.color / hr, 2)
        res = const * d2_const * d3_const
        return res
