from skimage import io, color
from scipy import ndimage
from scipy.ndimage import filters
from math import pow
import numpy as np
import matplotlib.pyplot as plt
import cv2

def lab_saliency(file_name, size=3, display_result=False):
    """The function calculates lab color space based(Frequency-tuned) saliency
of input rgb_image

Parameters
----------
rgb_image: M x N x 3 array (assumed to be color image in rgb space)
Input rgb image whose saliency map has to be calculated
size: scalar value, optional
size of the gaussian filter kernel to smooth the rgb_image
Returns
-------
Outputs: 2D array(M x N)
The functions returns the saliency map of Input Image
Number of rows and Cols. of input is same as input image
References
----------
Achanta,S. Hemami,F. Estrada,S. Susstrunk
'Frequency-tuned Salient region detection'
IEEE International Conference on Computer Vision and Pattern Recognition (CVPR 2009),
"""
    rgb_image = io.imread(file_name)
    #smooth the input rgb_image
    rgb_image = filters.gaussian_filter(rgb_image, size)
    #convert to lab color space
    lab_image = color.rgb2lab(rgb_image)
    mean = np.asarray([lab_image[:, :, 0].mean(), lab_image[:, :, 1].mean(), lab_image[:, :, 2].mean()])
    mean_subtracted = (lab_image - mean)**2
    saliency_map = mean_subtracted[:, :, 0] + mean_subtracted[:, :, 1] + mean_subtracted[:, :, 2]
    if(display_result):
        plt.imshow(saliency_map)
        plt.show()
    return saliency_map


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


def msss_saliency(file_name, size=5):
    rgb = io.imread(file_name)
    
    #gfrgb = ndimage.gaussian_filter(rgb, 3, mode='mirror')
    gfrgb = filters.gaussian_filter(rgb, size)
    # gfrgb = cv2.GaussianBlur(rgb, (3,3), 5)
    #gfrgb = ndimage.gaussian_filter(rgb, 3)
     
    lab = color.rgb2lab(gfrgb)
    
    height, width, dim = lab.shape
    
    l = lab[:,:,0]
    # lm = np.mean(l)
    
    a = lab[:,:,1]
    # am = np.mean(a)
    
    b = lab[:,:,2]
    # bm = np.mean(b)
    
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


def minmaxnormalization(vector):
    """
Makes the min max normalization over a numpy vector
$$ v_i = (v_i - min(v)) / max(v)
"""
    vector = vector - (vector.min())
    vector = vector / (vector.max() - vector.min())
    return vector

    

def frequency_tuned_saliency(img_src):
    """
Frequency Tuned Saliency.
Find the Euclidean distance between
the Lab pixel vector in a Gaussian filtered image
with the average Lab vector for the input image.
R. Achanta, S. Hemami, F. Estrada and S. Susstrunk,
Frequency-tuned Salient Region
Detection, IEEE International Conference on Computer
Vision and Pattern Recognition

Args:
image (np.array): an image.

Returns:
a 2d image saliency map.
"""
    image = cv2.imread(img_src)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #mean of each channel
    means = []
    for c in range(image.shape[2]):
        means.append(image[:, :, c].mean())
    means = np.asarray(means)

    image = cv2.medianBlur(image, 9)
    dist = (image - means) ** 2
    print("mean color is %s" % means)
    salmap = np.zeros((dist.shape[0], dist.shape[1]))
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            salmap[i][j] = np.sqrt(dist[i][j].sum())

    return minmaxnormalization(salmap)    
