import numpy as np


class Cluster:
    '''
    TEsted and working
    '''
    list_pixel = np.zeros(5)
    avg = np.array([0, 0, 0])
    # PIXEL_SIZE = 5
    COLOR_SIZE = 3
    PIXEL_SIZE = 5
    merged = False

    def __init__(self, np_pixel):
        # print("pixel ", pixel)
        self.list_pixel = np.array([np_pixel])
        self.avg = np_pixel[2:]

    def add_pixel(self, pixel):
        '''
        add pixel to cluster and set the avg of the cluster
        :param pixel: pixel to add
        :return: 0
        '''
        self.avg_with(pixel[2:])
        self.list_pixel = np.append(self.list_pixel, pixel)
        y = self.list_pixel.size / self.PIXEL_SIZE
        self.list_pixel = np.reshape(self.list_pixel, (y, self.PIXEL_SIZE))
        return 0

    def avg_with(self, pixel):
        '''
        set the avg color of the cluster with the new pixel
        :param pixel: pixel to add to cluster
        :return: 0
        '''
        avg = self.avg
        nb = self.list_pixel.size / self.COLOR_SIZE
        new_avg = np.zeros(self.COLOR_SIZE)
        for i in range(self.COLOR_SIZE):
            # print(i)
            # print(avg)
            # print(pixel)
            num = nb * avg[i] + pixel[i]
            new_avg[i] = num / (nb + 1)
        self.avg = new_avg
        return 0

    def equals(self, other):
        '''

        Determinate if self and other cluster are the same

        :param other: othe object cluster
        :return: boolean
        '''
        if (not (np.array_equal(self.avg, other.avg))):
            return False
        if (self.list_pixel.size != other.list_pixel.size):
            return False
        else:
            for pixel_ind in range(int(self.list_pixel.size / self.PIXEL_SIZE)):
                # print("this", self.list_pixel)
                # print("other", other.list_pixel)
                if (not (np.array_equal(self.list_pixel[pixel_ind], other.list_pixel[pixel_ind]))):
                    return False
        return True

    def is_neighbors(self, other):
        '''

        Tells if self and other are neighbors. If the two clusters have as least two pixel side by side, return true
        else, return false

        :param other: other clusterr
        :return: true if neighbors, else false
        '''
        for self_pixel in self.list_pixel:
            for other_pixel in other.list_pixel:
                # print(self.list_pixel)
                # print(other.list_pixel)
                # print(other_pixel)
                self_x = self_pixel[0]
                self_y = self_pixel[1]
                other_x = other_pixel[0]
                other_y = other_pixel[1]
                # print("x", self_x, "y", self_x)
                # print("x", other_x, "y", other_y)
                if (self_x == other_x):
                    if ((self_y == other_y + 1) or (self_y == other_y - 1)):
                        return True
                if (self_y == other_y):
                    if ((self_x == other_x + 1) or (self_x == other_x - 1)):
                        return True
        return False

    def add_cluster(self, cluster):
        '''
        add cluster to this cluster
        :param cluster: object cluster having a list of pixel
        :return: this after merge
        '''
        for pixel in cluster.list_pixel:
            self.add_pixel(pixel)
        cluster.merged = True
        return self

    def get_avg(self):
        return self.avg

    def get_pixel(self, index):
        return self.list_pixel[index]

    def can_merge(self, other, hr):
        # if (np.linalg.norm(self.avg[2:] - other.avg[2:], 2) < hr):
        if (np.linalg.norm(self.avg - other.avg, 2) < hr):
            # print("CAN MERGE")
            return True
        return False

    def get_size(self):
        return self.list_pixel.size

    def get_pixels(self):
        return self.list_pixel

#
# pixel = np.array([0, 0, 0, 0, 0])
# pixel3 = np.array([0, 1, 0, 0, 0])
# pixel4 = np.array([1, 0, 0, 0, 0])
# clt3 = Cluster(pixel3)
# clt = Cluster(pixel)
# clt.add_pixel(np.array([1, 1, 1, 1, 1]))
# pixel2 = np.array([255, 255, 255, 255, 255])
# clt2 = Cluster(pixel2)
# clt2.add_pixel(np.array([244, 244, 244, 244, 244]))
# clt.add_cluster(clt2)
#
# print(clt.list_pixel)
# print(clt.avg)
# print(clt2.list_pixel)
# print(clt3.is_neighbors(clt))
