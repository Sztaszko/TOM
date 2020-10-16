import numpy as np
from scipy import ndimage
from skimage import color, io
import time
import matplotlib.pyplot as plt
import skimage.measure as measure
import queue # Do implementacji lokalnej wersji rozrostu obszarów

def normalize(image):
    return (image - np.min(image))/(np.max(image) - np.min(image))

image  = normalize(color.rgb2gray(io.imread('CT_lungs.png')))
plt.figure()
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title("Image shape: " + str(image.shape))
plt.show()

def generate_circle(y_size, x_size, x_origin, y_origin, radius):
    image = np.zeros((y_size, x_size))
    x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
    indices = np.square((x_grid - x_origin)) + np.square((y_grid-y_origin)) < radius*radius
    image[indices] = 1
    return image

y_size = 1024
x_size = 1024
circle = generate_circle(y_size, x_size, 512, 512, 300)
circle = circle + (np.random.random((y_size, x_size))-0.5)*0.4
circle = normalize(circle)

def image_threshold(image, lower, upper):
    img_out=np.zeros(image.shape)
    object_point=np.bitwise_and(image>lower, image<upper)
    img_out[object_point]=1
    return img_out

def region_growing_local(image, seed, bottom_threshold, upper_threshold):
    pixel_queue = queue.Queue()
    visited = set()
    def get_neighbours(coordinate):
        x=coordinate[0]
        y=coordinate[1]
        neighbours=[(x+1,y),(x-1,y),(x,y+1),(x,y-1),(x+1,y-1),(x-1,y-1),(x+1,y+1),(x-1,y+1)]
        return neighbours
    segmentation_image = np.full(np.shape(image), False)
    pixel_queue.put(seed)
    visited.add(seed)
    while not pixel_queue.empty():
        current_pixel=pixel_queue.get()
        neighbours=get_neighbours(current_pixel) 
        for i in neighbours:
            j=set()
            j.add(i)
            if(np.bitwise_and(image[i]>image[seed]-bottom_threshold, image[i]<image[seed]+upper_threshold) and not j.issubset(visited)):
                segmentation_image[i]=True
                pixel_queue.put(i)
                visited.add(i)
            print("queue size:", pixel_queue.qsize())
        pass
    return segmentation_image


get_center = lambda image: (int(image.shape[0] / 2) - 1, int(image.shape[1] / 2) - 1)
circle_region_local = region_growing_local(circle, get_center(circle),  0.1, 0.1)
image_region_local = region_growing_local(image, get_center(image), 0.1, 0.1)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(circle, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(circle_region_local, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(image_region_local, cmap='gray')
plt.axis('off')
plt.show()

