from scipy.signal import convolve2d
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2

image=img.imread("Xray-300x247.jpg")
image_gray = color.rgb2gray(image)


S1=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
S2=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

Gx=convolve2d(image_gray,S1)
Gy=convolve2d(image_gray,S2)

magnitude=np.hypot(Gx,Gy)
phase=np.arctan(Gx/Gy) 
#wystepuje błąd dzielenia przez 0 - zostawione w celu zobrazowania roznicy pomiedzy
#wlasnym dzieleniem a funkcją z biblioteki cv2. Własciwy kąt to phase_cv2
phase_cv=cv2.phase(Gx,Gy,True)

plt.subplot(2,3,1)
plt.imshow(image_gray,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(2,3,2)
plt.imshow(Gx,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(Gy,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
plt.subplot(2,3,4)
plt.imshow(magnitude,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(2,3,5)
plt.imshow(phase)
plt.axis('off')
#dla porownania
plt.subplot(2,3,6)
plt.imshow(phase_cv)
plt.axis('off')
plt.show()