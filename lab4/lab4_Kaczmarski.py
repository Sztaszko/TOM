import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import signal
import cv2

im=plt.imread('Xray-300x247.jpg')
print(im.shape)
im_gray=skimage.color.rgb2gray(im)
im_fft=np.fft.fft2(im_gray)

def showFFT(fft):
    plt.imshow(np.log10(abs(fft)), cmap='gray')
    plt.axis('off')
    plt.show()
def showImage(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    

noise=np.random.normal(loc=0,scale=0.1,size=im_gray.shape)
im_noisy=im_gray+noise
(x,y)=im_noisy.shape
for i in range(x):
    for j in range(y):
        if im_noisy[i][j]<0:
            im_noisy[i][j]=0
        if im_noisy[i][j]>1:
            im_noisy[i][j]=1
showImage(im_noisy)
im_noisy_fft=np.fft.fft2(im_noisy)

mask = np.ones((5,5))/25
#im_filtered = signal.convolve2d(im_noisy, mask)
#im_filtered_fft = np.fft.fft2(im_filtered)
im_filtered = cv2.filter2D(im_noisy,-1,mask)
im_filtered_fft = np.fft.fft2(im_filtered)

plt.figure(figsize=(12,12))
plt.subplot(3,3,1)
plt.title("obraz poczatkowy")
plt.imshow(im_gray, cmap='gray')
plt.axis('off')
plt.subplot(3,3,2)
plt.title("obraz zaszumiony")
plt.imshow(im_noisy, cmap='gray')
plt.axis('off')
plt.subplot(3,3,3)
plt.title("obraz odszumiony")
plt.imshow(im_filtered, cmap='gray')
plt.axis('off')
plt.subplot(3,3,4)
plt.hist(im_gray.ravel(),bins=128, cumulative=True)
plt.title("Histogram oryginalny")
plt.subplot(3,3,5)
plt.hist(im_noisy.ravel(),bins=128, cumulative=True)
plt.title("Histogram zaszumiony")
plt.subplot(3,3,6)
plt.hist(im_filtered.ravel(),bins=128, cumulative=True)
plt.title("Histogram odszumiony")
plt.subplot(3,3,7)
plt.title("FFT oryginalny")
plt.imshow(np.log10(abs(im_fft)), cmap='gray')
plt.axis('off')
plt.subplot(3,3,8)
plt.title("FFT zaszumiony")
plt.imshow(np.log10(abs(im_noisy_fft)), cmap='gray')
plt.axis('off')
plt.subplot(3,3,9)
plt.title("FFT odszumiony")
plt.imshow(np.log10(abs(im_filtered_fft)), cmap='gray')
plt.axis('off')
plt.show()

plt.figure(figsize=(12,12))
plt.subplot(1,2,1)
plt.imshow(im_gray-im_noisy, cmap='gray')
plt.title("Roznica oryginal-zaszumiony")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(np.subtract(im_gray,im_filtered), cmap='gray')
plt.title("Roznica oryginal-odfiltrowany")
plt.axis('off')
plt.show()