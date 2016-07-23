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
        self.img_size = img.size
        pixel_size = clusters[0][0].size
        y_max = int(clusters[0].size / pixel_size)
        x_max = int(clusters.size / y_max / pixel_size)
        for x in range(x_max):
            for y in range(y_max):
                xy = np.array([x, y])
                xycolor = np.append(xy, clusters[x][y])
                self.list_cluster = np.append(self.list_cluster, Cluster.Cluster(xycolor))
        self.list_cluster = np.delete(self.list_cluster, 0)

    def run_segmentation(self, hr, nb_iter):
        new = np.array([])
        for iter in range(nb_iter):
            for fir_clust_ind in range(len(self.list_cluster)):
                print(fir_clust_ind)
                for sec_clust_ind in range(len(self.list_cluster)):
                    # print(self.list_cluster[fir_clust_ind].avg)
                    # print(self.list_cluster[fir_clust_ind].list_pixel)
                    if (not (self.list_cluster[sec_clust_ind].merged)):
                        if (not (self.list_cluster[fir_clust_ind].equals(self.list_cluster[sec_clust_ind]))):
                            if (self.list_cluster[fir_clust_ind].is_neighbors(self.list_cluster[sec_clust_ind])):
                                if (self.list_cluster[fir_clust_ind].can_merge(self.list_cluster[sec_clust_ind], hr)):
                                    self.list_cluster[fir_clust_ind].add_cluster(self.list_cluster[sec_clust_ind])
                                    # self.list_cluster = np.delete(self.list_cluster,sec_clust_ind)

    def create_segmented_image(self):
        img = Image.new("RGB", self.img_size)
        for clust in self.list_cluster:
            for pixel in clust.list_pixel:
                pixel[2:] = clust.get_avg()
                #        pupixel needs xy, but cluser doesnt have the pixel postion
                # print(clust.avg)
                # print(np.round(clust.avg))
                # print((pixel[1], pixel[0]), tuple(np.round(clust.avg)))
                img.putpixel((pixel[1], pixel[0]), tuple(np.round(clust.avg).astype(int)))
        return img

    def get_list_cluster(self):
        return self.list_cluster

    def get_clusters_det(self):
        for clust in self.list_cluster:
            print("self.list_cluster SIZE", self.list_cluster.size)
            print("clust", clust)
            print("size", clust.get_size())
            print("avg", clust6.get_avg())
            print("pizels", clust.get_pixels())


# epsi =.1
# neighbors =50
hr = 50
# hs=10
#
img = Image.open("images/fruit_filtered")
# msf = MSF.MSF(img, epsi, neighbors, hr, hs)
# new = msf.run_mean_shift()
# new.show()
seg = Segmentation(img)
seg.run_segmentation(hr, 6)
seg.get_clusters_det()
seg.create_segmented_image().show()
