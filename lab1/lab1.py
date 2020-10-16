import matplotlib.pyplot as plt
import matplotlib.image as img
import skimage
import skimage.color
import numpy as np

xray=img.imread('rtg.jpg')
xray_gray=skimage.color.rgb2gray(xray)

def negation(image):
    image=image-255
    return image

def logarytm(img):
    img=(np.log10(1+img))
    return img

plt.subplot(2,2,1)
plt.imshow(xray)

plt.subplot(2,2,2)
plt.imshow(xray_gray,'gray')

plt.subplot(2,2,3)
plt.imshow(negation(xray))

plt.subplot(2,2,4)
plt.imshow(logarytm(xray_gray))

plt.show()