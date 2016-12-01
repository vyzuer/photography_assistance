from skimage import io, img_as_float
from scipy import fftpack, ndimage, misc
from scipy.ndimage import uniform_filter, filters
from skimage.color import rgb2gray, rgb2lab
import numpy as np

from saliency_map import SaliencyMap
from saliency_achanta import saliency_achanta, msss_saliency, lab_saliency, frequency_tuned_saliency
from utils import OpencvIo
import time
import msss_saliency as msss

class Saliency:
    def __init__(self, image, saliency_type = 0, filter_size=3, mode="nearest", sigma=2.5):
        # saliency_type
        # 0 - spectral residual
        # 1 - Itti
        # 2 - Achanta 2008 fast mode
        # 3 - achanta 2010
        self.timer = time.time()
        if saliency_type == 0:
            self.saliency_map = self.__spectral_residual_saliency_map(image, filter_size, mode, sigma)
        elif saliency_type == 1:
            self.saliency_map = self.__itti_saliency_map(image)
        elif saliency_type == 2:
            # self.saliency_map = saliency_achanta(image_src)
            # self.saliency_map = lab_saliency(image_src)
            self.saliency_map = frequency_tuned_saliency(image)
        else:
            self.saliency_map = self.__find_saliency(image)

    def __find_saliency(self, image, size=5):
        gfrgb = filters.gaussian_filter(image, size)
         
        lab = rgb2lab(gfrgb)

        lab = np.ascontiguousarray(lab)

        saliency_map = msss.msss_saliency(lab)

        return saliency_map


    def __set_timer(self):
        self.timer = time.time()

    def __print_timer(self):
        print time.time() - self.timer
    def getSaliencyMap(self):
        return self.saliency_map
        
    def __spectral_residual_saliency_map(self, image_src, filter_size, mode, sigma):
        # Spectral Residual
        image = io.imread(image_src)
        image = img_as_float(rgb2gray(image))
        fft = fftpack.fft2(image)
        logAmplitude = np.log(np.abs(fft))
        phase = np.angle(fft)
        avgLogAmp = uniform_filter(logAmplitude, size=filter_size, mode=mode) 
        spectralResidual = logAmplitude - avgLogAmp
        sm = np.abs(fftpack.ifft2(np.exp(spectralResidual + 1j * phase))) ** 2

        # After Effect
        saliencyMap = ndimage.gaussian_filter(sm, sigma=sigma)

        return saliencyMap

    def __itti_saliency_map(self, image_src):
        self.__set_timer()
        image = io.imread(image_src)
        print "itti saliency started..."
        s_map = SaliencyMap(image)
        self.__print_timer()
        print "itti saliency done."

        return s_map.map

