from _hashlib import new
from matplotlib.pyplot import flag
import Kernel as K
from PIL import Image, ImageFilter
import scipy.spatial as sp
import numpy as np
import math
import decimal
import time
import pixel as pi


class MSF:
    kernels = np.array  # 2d array of kernel
    PIXEL_DIMENSION = 5
    dim5data = np.array
    image = 0
    epsi = 0
    neighbors = 0
    hs = 0
    hr = 0
    KD_tree = 0

    '''
    size
    '''

    def __init__(self, img, epsi, neighbors, hs, hr):
        '''

        :param img:
        :return:
        '''
        # self.image = Image.open(path)
        self.image = img
        self.epsi = epsi
        self.neighbors = neighbors
        self.hs = hs
        self.hr = hr
        larg = self.image.size[0]
        alt = self.image.size[1]
        # dt = np.dtype(K.Kernel)
        # self.kernels = np.zeros((larg, alt), dt)
        self.dim5data = np.zeros((larg * alt, 5))
        print(self.dim5data)
        data = self.image.getdata()
        for y in range(alt * larg):
            # for x in range(larg):
            color = data[y]
            # self.kernels[x, y] = K.Kernel((x, y), color)
            # self.dim5data[x, y] = [x, y, color[0], color[1], color[2]]
            self.dim5data[y] = [y // larg, y % alt, color[0], color[1], color[2]]

    # print(int(self.dim5data[0].size / self.PIXEL_DIMENSION),
    #       int(self.dim5data.size / self.dim5data[0].size))

    def run_mean_shift(self):
        '''
        run the mean shift algorithm
        '''
        # newshape = self.dim5data.size / self.PIXEL_DIMENSION
        # cpy = self.dim5data, (newshape, self.PIXEL_DIMENSION))  # shape  larg*alt/pixel5d , 5
        cpy = self.dim5data  # shape  larg*alt/pixel5d , 5 OOPIA QUE RECEBERÁ NOVOS VALORES MAS QUE TERÁ OS PIXELS REMOVIDOS QUANDO CONVERGIREM
        cpy_bis = np.copy(cpy)  # copia dos pixels que vai servir para a KDTREE
        cpy2 = np.copy(cpy)  # copia que receberão os novos valores
        self.KD_tree = sp.KDTree(cpy_bis)
        old_size = cpy.size
        while (cpy.size > 0):  # enquanto tiver ponrtos que não convergiram
            print(cpy.size)
            ind_cpy = 0
            ind_cpy2 = 0
            while (ind_cpy < (
                        cpy.size / self.PIXEL_DIMENSION)):  # enquanto não tiver percorrido toda a lista de pontos da imagem
                this_pixel = cpy[ind_cpy]

                nnL = self.get_nearest_ng(cpy_bis, this_pixel, self.neighbors)
                new_pixel = self.calculate_new_pixel(nnL, this_pixel)
                dist_vector = np.linalg.norm(pi.Pixel.diff_normal(new_pixel, this_pixel), 2)
                # print(new_pixel)
                cpy2[ind_cpy2] = new_pixel
                if (dist_vector < self.epsi):
                    cpy = np.delete(cpy, ind_cpy, 0)
                    ind_cpy -= 1
                else:
                    cpy[ind_cpy] = new_pixel
                ind_cpy += 1
                ind_cpy2 += 1
        mat_filtered_image = pi.Pixel.get_rgb(cpy2)
        print("mat_filtered_image s", mat_filtered_image[10 * self.image.size[0] + 10])
        new_image = self.create_new_image(mat_filtered_image, self.image.size,
                                          "RGB")
        print("new_image", new_image.getpixel((10, 10)))
        print("true image", self.image.getpixel((10, 10)))

        return new_image

    def calculate_new_pixel(self, ng_arr, pixel):
        num = np.array([0, 0, 0, 0, 0])
        denum = 0
        for point in ng_arr:  # para todos os vizinhos encontrados
            xs = point[:2]
            xr = point[2:]
            cs = math.exp(
                -1 * np.linalg.norm(pi.Pixel.mult_by_const(pi.Pixel.diff_normal(xs, pixel[:2]), 1.0 / self.hs), 2))
            cr = math.exp(
                -1 * np.linalg.norm(pi.Pixel.mult_by_const(pi.Pixel.diff_normal(xr, pixel[2:]), 1.0 / self.hr), 2))
            # print(cs, cr)
            # print("cs ", cs)
            # print("cr ", cr)
            # print("point ", point)
            num = pi.Pixel.sum_normal(num, pi.Pixel.mult_by_const(point, cs * cr))
            # print("num ", num)
            denum += cs * cr
            # print("denum ", denum)
        new_pixel = pi.Pixel.mult_by_const(num, 1.0 / denum)
        # new_pixel = self.round(new_pixel)  # fazer śo no final para os valores RGB

        return new_pixel

    def create_new_image(self, data, xy, mode):
        '''
        create a new image using data sequence where each element is a pixel of the image

        :param data: np.array of np.array's where the latest are the RGB pixels
        :param x: width of the new image
        :param y: height of the new image
        :param mode: mode of the new image
        :return: the new image
        '''
        new = Image.new(mode, xy)
        data = self.np_array_to_tuple(data)
        print(data)
        new.putdata(data)
        return new

    def np_array_to_tuple(self, arr):
        for i in range(int(arr.size / 3)):
            arr[i] = tuple(arr[i])
            print(tuple(arr[i]))
        return arr

    def get_nearest_ng(self, data, pixel, nb):
        '''
        Get nb pixels which are closest to pixel in d-dimensions.

        :param data: list of pixels
        :param pixel: pixel which we want to find neighbors
        :param nb: number of neighbors
        :return: the list of closest pixels
        '''
        dim = data.shape
        d, i = self.KD_tree.query(pixel, nb)
        l = []
        for k in i:
            l.append(data[k])
        return l

    def test(self):
        zero9 = np.zeros(9, tuple)
        zero9[4] = (255, 255, 255)
        # print(zero9)
        img = Image.new("RGB", (3, 3))
        img.putdata(zero9)
        msf = MSF(img, 100, 9, 10, 10)
        msf.run_mean_shift().show()

    def unblockshaped(self, arr, h, w):
        '''
        TESTED AND WORKING
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        '''
        n, nrows, ncols = arr.shape
        return (arr.reshape(h // nrows, -1, nrows, ncols)
                .swapaxes(1, 2)
                .reshape(h, w))


img = Image.open("images/fruit_64.png")
msf = MSF(img, 1, 10, 10, 10)
new = msf.run_mean_shift()
msf2 = MSF(img, .1, 2, 10, 10)
new2 = msf2.run_mean_shift()
# new2.show()
new2.show()
print(new2.getpixel((10, 10)))
print(new.getpixel((10, 10)))
new.show()
img.show()
