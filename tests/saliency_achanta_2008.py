from skimage import io, color
from scipy import ndimage
from math import pow
import numpy as np
import matplotlib.pyplot as plt
import cv2

def saliency_achanta(file_name):

    rgb = io.imread(file_name)
    
    #gfrgb = ndimage.gaussian_filter(rgb, 3)
    gfrgb = cv2.GaussianBlur(rgb, (3,3), 3)
     
    lab = color.rgb2lab(gfrgb)
    
    h, w, d = lab.shape
    
    l = lab[:,:,0]
    lm = np.mean(l)
    
    a = lab[:,:,1]
    am = np.mean(a)
    
    b = lab[:,:,2]
    bm = np.mean(b)
    
    sm = (l-lm)**2 + (a-am)**2 + (b-bm)**2
    plt.imshow(sm)
    plt.show()

file_name = "1.jpg"    
msss_saliency(file_name)    
