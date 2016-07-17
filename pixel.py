import numpy as np


class Pixel:
    dimension = 5

    def __init__(self, dimension):
        self.dimension = dimension

    @staticmethod
    def get_rgb(arr):
        '''
        get the rgb value of a 5D pixel where position 2,3 and 4 are related to RGB
        :param arr:  5d array where 0, 1 are x and y, 2,3 and 5, R,G and B respectively
        :return: np.array([R,G,B])
        '''
        nb_pixel = 5
        data = np.zeros(arr.size / nb_pixel, tuple)
        # data = np.zeros(arr.size, tuple)
        for i in range(len(arr)):
            data[i] = (int(arr[i][2]), int(arr[i][3]), int(arr[i][4]))
        print("data.size")
        print(data.size)
        return data

    # @staticmethod
    # def get_rgb2(arr):
    #     '''
    #     get the rgb value of a 5D pixel where position 2,3 and 4 are related to RGB
    #     :param arr:  5d array where 0, 1 are x and y, 2,3 and 5, R,G and B respectively
    #     :return: np.array([R,G,B])
    #     '''
    #     print(arr.size)
    #     data = np.zeros(3 * arr.size / 5, tuple)
    #     newshape = arr.size / 5
    #     data = np.reshape(data, (newshape, 3))
    #     # data = np.zeros(arr.size, tuple)
    #     for i in range(len(arr)):
    #         data[i] = tuple(self.image.getpixel((arr[i][0], arr[i][1])))
    #         # data[i] = (int(arr[i][2]), int(arr[++++++++++++++++i][3]), int(arr[i][4]))
    #         print("data.size")
    #         print(data.size)
    #     return data

    @staticmethod
    def round(pixel):
        '''
        Round the value of the pixel
        :param pixel: np.array([R,G,B])
        :return: np.array([round(R),round(G),round(B)])
        '''
        new_pixel = np.zeros(pixel.size)
        for i in range(len(pixel)):
            new_pixel[i] = round(pixel[i])
        return new_pixel

    @staticmethod
    def diff(x_pixel, y_pixel):
        '''

        Makes the subtraction between two pixels in RGB mode.
        If a value of a band is lower than 0, then it is set to 0

        :param x_pixel: minuend pixel
        :param y_pixel: subtrahend pixel
        :return: the subtraction of x_pixel and y_pixel
        '''
        nb_pixel = 5
        new = np.zeros(nb_pixel)
        for ind in range(nb_pixel):
            diff = x_pixel[ind] - y_pixel[ind]
            new[ind] = 0 if (diff < 0 and ind > 1) else diff
        return new

    @staticmethod
    def diff_normal(x_pixel, y_pixel):
        '''

        Makes the subtraction between two pixels in RGB mode.

        :param x_pixel: minuend pixel
        :param y_pixel: subtrahend pixel
        :return: the subtraction of x_pixel and y_pixel
        '''
        new = np.zeros(x_pixel.size)
        for ind in range(x_pixel.size):
            diff = x_pixel[ind] - y_pixel[ind]
            new[ind] = diff
        return new

    @staticmethod
    def index_2d_to_1d(x,y):
        return
    @staticmethod
    def sum(x_pixel, y_pixel):
        '''

        Sum two pixels up in RGB mode
        If a value of a band is bigger than 255, then it is set to 255

        :param x_pixel:
        :param y_pixel:
        :return: sum between the two parameters
        '''
        new = np.zeros(nb_pixel)
        for ind in range(nb_pixel):
            sum = x_pixel[ind] + y_pixel[ind]
            new[ind] = 255 if (sum > 255 and ind > 1) else sum
        return new

    @staticmethod
    def sum_normal(x_pixel, y_pixel):
        '''
        Sum pixels values.

        :param x_pixel:
        :param y_pixel:
        :return:
        '''
        nb_pixel = 5
        new = np.zeros(nb_pixel)
        for ind in range(nb_pixel):
            sum = x_pixel[ind] + y_pixel[ind]
            new[ind] = sum
        return new

    @staticmethod
    def mult_by_const(vector, scalar):
        '''
        Multiply each value of the vector by a scalar

        :param vector: numeric array
        :param scalar: numeric value
        :return: new vector with values multiplied by scalar
        '''
        new = np.zeros(vector.size)
        for i in range(vector.size):
            new[i] = vector[i] * scalar
        return new
