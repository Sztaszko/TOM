import numpy as np
import matplotlib.pylab as plt
from skimage import color, morphology
from scipy import signal
import scipy


#kopia z notatnika Jupyter

im = plt.imread('Xray-300x247.jpg')
im.shape
im_gray = color.rgb2gray(im)

def getBinaryImage(img, threshold, mode='upper'):
    if mode == 'upper':
        return im_gray > threshold
    elif mode == 'lower':
        return im_gray < threshold
    else:
        raise Exception("Mode not supported")
        
def compareImages(img1, img2):
    plt.figure(figsize=(12,12))
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('processed')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    if (img1.dtype == bool and img2.dtype == bool):
        plt.imshow(np.asarray(img2, dtype=np.uint16) - np.asarray(img1, dtype=np.uint16), cmap='gray')
    else:
        plt.imshow(img2 - img1, cmap='gray')
    plt.title('difference')
    plt.axis('off')
    

def isInCircle(x, y, center_x, center_y, radius):
    out = (x - center_x)**2 + (y - center_y)**2 < radius**2
    return out

def createDisk(radius, SE):
    disk = np.zeros_like(SE)
    x_SE, y_SE = np.shape(SE)
    x_center = int(np.floor(x_SE / 2))
    y_center = int(np.floor(y_SE / 2))
    for i in range(0,np.shape(SE)[0]):
        for j in range(0, np.shape(SE)[1]):
            disk[i,j] = isInCircle(i, j, x_center, y_center, radius + 0.2)
    return disk

def createSE(size = (3,3), shape='filled', radius = -1):
    SE = np.ones(size, np.uint8)
    if shape == 'filled':
        return SE
    elif shape == 'disk':
        if radius == -1:
            radius = int(np.floor(np.max(np.shape(SE)) / 2))
        return createDisk(radius, SE)

def getImageSubset(img, i, j, SE, x_SE, y_SE):
    subset = []
    lowerBound_X = i - int(np.floor(x_SE / 2))
    upperBound_X = i + int(np.floor(x_SE / 2))
    lowerBound_Y = j - int(np.floor(y_SE / 2))
    upperBound_Y = j + int(np.floor(y_SE / 2))
    x_max, y_max = np.shape(img)
    for se_x, k in enumerate(range(lowerBound_X, upperBound_X+1)): #enumerate - numeruje lowerBound_X od upperBound_X+1
        for se_y, l in enumerate(range(lowerBound_Y, upperBound_Y+1)):
            if k < 0 or l < 0 or k > x_max or l > y_max: #jak jest poza obrazem
                subset.append(0)
            elif k < x_max and l < y_max: #jak jest w obrazie
                subset.append(img[k,l] * SE[se_x - 1, se_y - 1]) #to dodaj do subset iloczyn
    return subset

def dilateManually(img, SE = np.ones((3,3), np.uint8)):
    x_SE, y_SE = np.shape(SE)
    x, y = np.shape(img)
    img_dilate = np.zeros_like(img)
    for i in range(0, x - 1):
        for j in range(0, y - 1):
            img_dilate[i,j] = np.max(getImageSubset(img, i, j, SE, x_SE, y_SE))
    return img_dilate

#zad1 erozja zadanym SE
def eroseManually(img, SE=np.ones((3,3),np.uint8)):
    x_SE,y_SE=np.shape(SE)
    x,y=np.shape(img)
    img_erose=np.zeros_like(img)
    for i in range(0,x-1):
        for j in range(0,y-1):
            img_erose[i,j]=np.min(getImageSubset(img,i,j,SE,x_SE,y_SE))
    return img_erose
    

im_binary=getBinaryImage(im_gray,np.mean(im_gray))

compareImages(im_gray, im_binary)
SE = createSE(size = (5,5))
im_erose_function = eroseManually(im_gray, SE)
im_erose_lib = morphology.erosion(im_gray,SE)

im_binary_erose_function = eroseManually(im_binary, SE)
im_binary_erose_lib = morphology.erosion(im_binary,SE)

compareImages(im_binary, im_binary_erose_function)
compareImages(im_binary, im_binary_erose_lib)

#zad3

def otwarcie(img,SE):
    subimg=eroseManually(img,SE)
    subimg=dilateManually(subimg,SE)
    return subimg

im_otwarcie=otwarcie(im_gray,SE)
compareImages(im_gray,im_otwarcie)

#zad4
def zamkniecie(img,SE):
    subimg=dilateManually(img,SE)
    subimg=eroseManually(subimg,SE)
    return subimg

im_zamkniecie=zamkniecie(im_gray,SE)
compareImages(im_gray,im_zamkniecie)


#zad5