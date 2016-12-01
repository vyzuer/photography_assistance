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

    return sm

def iisum(iimg,x1,y1,x2,y2):
    sum = 0
    if(x1>1 and y1>1) :
        sum = iimg[y2-1,x2-1]+iimg[y1-2,x1-2]-iimg[y1-2,x2-1]-iimg[y2-1,x1-2]
    elif(x1<=1 and y1>1) :
        sum = iimg[y2-1,x2-1]-iimg[y1-2,x2-1]
    elif(y1<=1 and x1>1) :
        sum = iimg[y2-1,x2-1]-iimg[y2-1,x1-2]
    else :
        sum = iimg[y2-1,x2-1]
    
    return sum


def msss_saliency(file_name):
    rgb = io.imread(file_name)
    
    #gfrgb = ndimage.gaussian_filter(rgb, 3, mode='mirror')
    gfrgb = cv2.GaussianBlur(rgb, (3,3), 3)
    #gfrgb = ndimage.gaussian_filter(rgb, 3)
     
    lab = color.rgb2lab(gfrgb)
    
    height, width, dim = lab.shape
    
    l = lab[:,:,0]
    lm = np.mean(l)
    
    a = lab[:,:,1]
    am = np.mean(a)
    
    b = lab[:,:,2]
    bm = np.mean(b)
    
    # create integral images
    li = np.cumsum(np.cumsum(l, axis=1), axis=0)
    ai = np.cumsum(np.cumsum(a, axis=1), axis=0)
    bi = np.cumsum(np.cumsum(b, axis=1), axis=0)

    sm = np.zeros(shape=(height, width))
    for j in range(1, height+1):
        yo = min(j, height-j)
        y1 = max(1,j-yo)
        y2 = min(j+yo,height)
        for k in range(1, width+1):
            xo = min(k,width-k)
            x1 = max(1,k-xo)
            x2 = min(k+xo,width)
            invarea = 1.0/((y2-y1+1)*(x2-x1+1))
            lm = iisum(li,x1,y1,x2,y2)*invarea
            am = iisum(ai,x1,y1,x2,y2)*invarea
            bm = iisum(bi,x1,y1,x2,y2)*invarea
            #---------------------------------------------------------
            # Compute the saliency map
            #---------------------------------------------------------
            sm[j-1,k-1] = (l[j-1,k-1]-lm)**2 + (a[j-1,k-1]-am)**2 + (b[j-1,k-1]-bm)**2

    img = (sm-np.min(sm))/(np.max(sm)-np.min(sm))

    return img

