import numpy as np
import matplotlib.pylab as plt
from skimage import color

def CheckIfBinary(img):
    if np.size(np.shape(im))==3:
        img=color.rgb2gray(img)
    w,h=np.shape(img)
    flaga=False
    for i in range(w):
        for j in range(h):
            if i ==0 or j==0 or i==1 or j==1:
                flaga=True
            else:
                flaga=False
    if flaga:
        return img
    else:
        return img > np.mean(img)



#z notatnika Jupyter
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

def erodeManually(img, SE=np.ones((3,3),np.uint8)):
    x_SE,y_SE=np.shape(SE)
    x,y=np.shape(img)
    img_erose=np.zeros_like(img)
    for i in range(0,x-1):
        for j in range(0,y-1):
            img_erose[i,j]=np.min(getImageSubset(img,i,j,SE,x_SE,y_SE))
    return img_erose
    

def getEdges(img,SE=np.ones((3,3)),mode="in"):
    img=CheckIfBinary(img)
    if mode=="in":
        return img^erodeManually(img,SE)
    elif mode=="out":
        return dilateManually(img,SE)^img
    else:
        raise Exception("Mode not supported")


im = plt.imread('Xray-300x247.jpg')
im_binary=CheckIfBinary(im)
plt.figure(figsize=(16,4))
plt.subplot(1,4,1)
plt.imshow(im)
plt.title("original")
plt.axis('off')
plt.subplot(1, 4, 2)
plt.imshow(im_binary, cmap='gray')
plt.title('binary image')
plt.axis('off')
    
SE=np.ones((3,3))
plt.subplot(1, 4, 3)
im_edges=getEdges(im,SE,"in")
plt.imshow(im_edges,cmap='gray')
plt.title('Edges in')
plt.axis('off')
plt.subplot(1, 4, 4)
im_edges=getEdges(im,SE,"out")
plt.imshow(im_edges,cmap='gray')
plt.title('Edges out')
plt.axis('off')