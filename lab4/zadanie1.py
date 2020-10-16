#zmiana working directory na folder zawierajacy zdjecie
import os 
os.chdir('/Users/komala/Documents/Python/zdjecia')

#importowanie uzywanych bibliotek
from matplotlib import pyplot as plt
import numpy as np
from skimage.color import rgb2gray 
from skimage.exposure import cumulative_distribution
import cv2

#funkcja do rysowania histogramu i obrazu
def createHistogram(image,number_of_bins=256,cumulative=False):
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(image,'gray')
    plt.xticks([]),plt.yticks([])
    plt.title('Picture')
    plt.subplot(2,1,2)
    plt.hist(np.ravel(image),bins = number_of_bins,cumulative = cumulative)
    plt.title('Histogram')
    plt.show()
    
#funkcja do pokazywania roznic pomiedzy obrazami 
def showDifference(compute_img,gray_image):
    difference = np.subtract(compute_img,gray_image)
    plt.figure()
    plt.imshow(difference,'gray')
    plt.title('Difference between images')
    
#funkcja cdf(cumulative distibution funcyion)
def cdf(image):
    c, b = cumulative_distribution(image) 
    c = np.insert(c, 0, [0]*b[0])
    c = np.append(c, [1]*(255-b[-1]))
    return c  


#funkcja do wyrownywania histogramu
def equal_hist(image,number_of_bins=256):   
    c = cdf(image)
    c_floor = np.floor(c*(number_of_bins-1))
    equal_img = (np.reshape(c_floor[image.ravel()],image.shape)).astype(np.uint8)
    return equal_img


#funkcja do dopasowywania histogramu
def match_hist(image_in,image_temp,number_of_bins=256):
    c = cdf(image_in)
    c_t = cdf(image_temp)
    pixels = np.arange(number_of_bins)
    new_pixels = np.interp(c, c_t, pixels) 
    match_img = (np.reshape(new_pixels[image_in.ravel()], image_in.shape)).astype(np.uint8)
    return match_img    


#gray picture - input
img = plt.imread('Xray.jpg')
img_gray = rgb2gray(img)
img_gray_u8 = np.asarray(img_gray*255, dtype=np.uint8)
createHistogram(img_gray_u8,cumulative=True)
#gray picture - template
imgt = plt.imread('CT_lungs.png')
imgt_gray = rgb2gray(imgt)
imgt_gray_u8 = np.asarray(imgt_gray*255, dtype=np.uint8)
createHistogram(imgt_gray_u8,cumulative=True)

#equaled picture
equal_img = equal_hist(img_gray_u8)
createHistogram(equal_img,cumulative=True)
#roznica pomiedzy obrazami
showDifference(equal_img,img_gray_u8)
#cv2 equal image - dodatek
img2 = cv2.imread('Xray.jpg',0)
equal_cv2 = cv2.equalizeHist(img2)

#matched picture
match_img = match_hist(img_gray_u8,imgt_gray_u8)
createHistogram(match_img,cumulative=True)
#roznica pomiedzy obrazami
showDifference(match_img,img_gray_u8)

