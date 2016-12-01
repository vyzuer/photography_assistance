from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

import SalientObject as sal_obj

def processImage(src, dump_path):
    my_obj = sal_obj.SalientObjectDetection(src)
    #my_obj.plotMaps()
    my_obj.processSegments(dump_path)
    #fp = open("test.fv", 'w')
    #my_obj.dumpFeatureVector(fp)
    #fp.close()
