import numpy as np
import math
import pixel as pi


class Kernel:
    org_pos=np.zeros(3)
    pixel = np.zeros(5)
    # position = np.zeros(2)  # [x,y]
    # color = np.zeros(3)  # [r,g,b]
    has_conv = False

    def __init__(self, pixel):
        self.org_pos=pixel[:2]
        self.pixel = pixel

    def has_converged(self, new_pixel, epsi):
        dist_vector = np.linalg.norm(new_pixel- self.pixel, 2)
        if (dist_vector < epsi):
            self.has_conv = True
            return True
        return False


    def move(self, position):
        self.pixel[:2] = position

    def set_color(self, color):
        self.pixel[2:] = color

    def get_color(self):
        return self.pixel[2:]

    def get_postion(self):
        return self.pixel[:2]

    @staticmethod
    def get_kernel_pos(arr_kernel, position):
        for kern in arr_kernel:
            if (kern.get_postion() == position):
                return kern
        return -1

    @staticmethod
    def set_kernel_col(arr_kernel, position, color):
        for kern in arr_kernel:
            if (kern.get_postion() == position):
                kern.set_color(color)
                return 0
        return -1

    def calculate_kernel(self, hs, hr):
        '''

        :param hs: constant de entrada para x,y  d=2
        :param hr: input constant for r,g,b d=3
        :return: K hs,hr
        '''
        c = 0
        const = c / (math.pow(hs, 2) * math.pow(hr, 3))
        d2_const = np.linalg.norm(self.get_postion() / hs, 2)
        d3_const = np.linalg.norm(self.get_color() / hr, 2)
        res = const * d2_const * d3_const
        return res
