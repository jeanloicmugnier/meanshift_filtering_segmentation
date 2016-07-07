import numpy as np
import math


class Color_Space:
    @staticmethod
    def convert_rgb_to_xyz(pixel):
        mat = np.array(
            [0.412453, 0.35758, 0.180423, 0.212671, 0.212671, 0.072169, 0.019334, 0.119193, 0.950227]).reshape((3, 3))
        # mat = np.array(
        #     [0, 0, 1, 0, 1, 0, 1, 0, 0]).reshape((3, 3))
        xyz = np.dot(mat, pixel)
        return xyz

    @staticmethod
    def convert_rgb_to_luv(pixel):
        '''
        TO BE TESTED
        converts the rgb pixel to a LUV pixel. Calls conver_rgb_to_xyz

        :param pixel: [r,g,b]
        :return: [L,U,V]
        '''
        xyz = Color_Space.convert_rgb_to_xyz(pixel)
        X = xyz[0]
        Y = xyz[1]
        Z = xyz[2]
        print("X " + str(X))
        print("Y " + str(Y))
        print("z " + str(Z))
        y = Y / 255.0
        epsi = 0.008856
        print("y " + str(y))
        if (y > epsi):
            L = (116. * (math.pow(y, 1.0 / 3))) - 16
        else:
            L = 903.3 * y
        if (X + (15. * Y) + (3. * Z) == 0):
            u1 = 4.0
            v1 = 9. / 15
        else:
            u1 = 4. * X / (X + (15. * Y) + (3. * Z))
            v1 = 9. * Y / (X + (15. * Y) + (3. * Z))
        print("L " + str(L))
        print("u1 " + str(u1))
        print("v1 " + str(v1))
        ur = 0.19784977571475
        vr = 0.46834507665248
        U = 13.0 * L * (u1 - ur)
        V = 13.0 * L * (v1 - vr)
        return np.array([L, U, V])
