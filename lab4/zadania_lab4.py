import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import cumulative_distribution
import skimage
import scipy

def my_hist(img, bin_num=256):
    hist=np.zeros(bin_num)
    for i in range(bin_num):
        for j in img.ravel():
            if j>(256/bin_num)*i and j<=((256/bin_num)*(i+1)-1):
                hist[i]=hist[i]+1
    return hist
    
            

im=plt.imread('Xray-300x247.jpg')
im_gray=skimage.color.rgb2gray(im)
print(im.shape)
print(my_hist(im_gray))

plt.figure()
plt.hist(im_gray.ravel(), bins=256)
plt.show()

