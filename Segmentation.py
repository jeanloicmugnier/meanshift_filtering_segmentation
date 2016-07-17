import MSF
import numpy as np
import Cluster
from PIL import Image


class Segmentation:
    '''
    in the beg, list of cluster = list of pixels of the imagess
    '''
    # list_cluster = np.array([Cluster.Cluster(np.array([0, 0, 0, 0, 0]))],dtype=Cluster.Cluster)
    list_cluster = 0
    img_size = 0

    def __init__(self, img):
        clusters = np.array(img)
        # print("clusters")
        # print(clusters)
        # print("clusters size")
        # print(clusters.size)
        self.img_size = img.size
        # print("self.img_size")
        # print(self.img_size)
        pixel_size = clusters[0][0].size
        y_max = int(clusters[0].size / pixel_size)
        x_max = int(clusters.size / y_max / pixel_size)
        # print("larg and width")
        # print(x_max)
        # print(y_max)
        # self.list_cluster = np.zeros((1, x_max * y_max, pixel_size), dtype=Cluster.Cluster)
        # print(self.list_cluster[0])
        # print(self.list_cluster[0][0])
        for x in range(x_max):
            for y in range(y_max):
                # print("x,y", x, y)
                # print("self.list_cluster[x][y]", self.list_cluster[x][y])
                # print("clusters[x][y]", clusters[x][y])
                self.list_cluster = np.append(self.list_cluster, Cluster.Cluster(clusters[x][y]))
        self.list_cluster = np.delete(self.list_cluster, 0)

    def run_segmentation(self, hr):
        for fir_clust_ind in range(len(self.list_cluster)):
            for sec_clust_ind in range(len(self.list_cluster)):
                if (not (self.list_cluster[fir_clust_ind].equals(self.list_cluster[sec_clust_ind]))):
                    # the two clusters aren't the same
                    if (self.list_cluster[fir_clust_ind].is_neighbors(self.list_cluster[sec_clust_ind])):
                        # the clusters are neinhors in 2D
                        if (self.list_cluster[fir_clust_ind].can_merge(self.list_cluster[sec_clust_ind], hr)):
                            # the clusters can merge RGB proximity
                            # add the sec cluter to the irst
                            self.list_cluster[fir_clust_ind].add_cluster(self.list_cluster[sec_clust_ind])
                            # remove the second cluster (merged one) from the list of clusters
                            np.delete(self.list_cluster[sec_clust_ind])

    def create_segmented_image(self):
        img = Image.new("RGB", self.img_size)
        for clust in self.list_cluster:
            for pixel in clust.get_pixel():
                pixel[2:] = clust.get_avg()
                img.putpixel((pixel[0], pixel[1]), pixel[2:])
        return img

    def get_list_cluster(self):
        return self.list_cluster

    def get_clusters_det(self):
        for clust in self.list_cluster:
            print("self.list_cluster SIZE", self.list_cluster.size)
            print("clust", clust)
            print("size", clust.get_size())
            print("avg", clust.get_avg())
            print("pizels", clust.get_pixels())


# epsi =.1
# neighbors =50
hr=10
# hs=10
#
img = Image.open("images/fruit_filtered")
# msf = MSF.MSF(img, epsi, neighbors, hr, hs)
# new = msf.run_mean_shift()
# new.show()
seg = Segmentation(img)
seg.run_segmentation(hr)
seg.get_clusters_det()
seg.create_segmented_image().show()
