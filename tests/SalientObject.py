from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.segmentation import relabel_sequential
from skimage import img_as_float
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import string
from scipy import ndimage
import os
import segmentation as seg
import Saliency as saliency
import time
import cv2
from skimage.feature import hog

NUM_OF_SALIENT_OBJECTS = 30
FV_LENGTH = 200

class SalientObjectDetection:
    def __init__(self, image_src, a_score=1):
        self.timer = time.time()
        self.image_src = image_src
        self.a_score = a_score
        self.image = io.imread(image_src)
        print "Image source : ", image_src

        #print "starting segmentation..."
        self.__set_timer("segmentation")
        segment_object = seg.SuperPixelSegmentation(self.image, seg_type=1, num_of_salient_objects=NUM_OF_SALIENT_OBJECTS)
        
        #print "segmentation done."
        self.segment_map = segment_object.getSegmentationMap()
        #np.savetxt("segment_map.txt", self.segment_map, '%d')
        self.slic_map = segment_object.getSlicSegmentationMap()
        #self.dummy_map = segment_object.getDummyMap()
        self.prominent_color_segment = segment_object.getProminentColorSegment()
        self.deleted_segment = segment_object.getDeletedSegmentID()
        self.__print_timer("segmentation")
        
        #print "starting saliency detection..."
        self.__set_timer("Saliency")
        saliency_object = saliency.Saliency(self.image_src, 3)
        
        #print "saliency done."
        self.saliency_map = saliency_object.getSaliencyMap()
        #np.savetxt("saliency.map", self.saliency_map, fmt='%-5.3f')
        #print self.saliency_map
        self.__print_timer("Saliency")

        #print "starting post processing..."
        self.__set_timer("post processing")
        self.saliency_list, self.salient_objects, self.pixel_count = self.__find_saliency_of_segments(self.segment_map, self.saliency_map)
        self.ave_color, self.positions = self.__find_features(self.segment_map, self.pixel_count, self.saliency_list, self.salient_objects)
        self.segment_map2 = self.__remove_non_salient_objects(self.salient_objects, self.segment_map)
        self.__print_timer("post processing")
        #print "post processing done."
    
    def processSegments(self, dump_path):
        smap, fwd, inv = relabel_sequential(self.segment_map2, offset=1)
        segs = ndimage.find_objects(smap)

        dir_name = os.path.split(self.image_src)[1]
        dir_name = os.path.splitext(dir_name)[0]

        dir_path = dump_path + dir_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        seg_path = dir_path + "/segments/"   
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        feature_file = dir_path + "/feature.list"
        saliency_file = dir_path + "/saliency.list"

        fp = open(saliency_file, 'w')
        fp1 = open(feature_file, 'w')

        self.__plot_maps(dir_path)

        for i in xrange(NUM_OF_SALIENT_OBJECTS):
            segment_id = self.saliency_list[i][1]
            
            fp.write("%0.6f\n" % self.saliency_list[i][0])

            new_seg_id = fwd[segment_id]
            segment_img = self.image[segs[new_seg_id-1]]
            segment_copy = np.copy(segment_img)

            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(8, 3, forward=True)
            plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)

            self.__dump_segment_features(segment_copy, fp1)

            mask = smap[segs[new_seg_id-1]]
            idx=(mask!=new_seg_id)
            segment_copy[idx] = 255, 255, 255

            ax[0].imshow(segment_img)
            ax[0].set_title("Image")
            ax[1].imshow(segment_copy)
            ax[1].set_title("Object")
            file_name = seg_path + str(i+1) + ".png"
            plt.savefig(file_name)

        fp.close()
        fp1.close()

    def __dump_segment_features(self, image_src, fp):
        fv = []
    
        surfHist = self.__xSurfHist(image_src)
        surfHist = np.asarray(surfHist)
        cv2.normalize(surfHist, surfHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(surfHist)

        hogHist = self.__xHOGHist(image_src)
        hogHist = np.asarray(hogHist)
        cv2.normalize(hogHist, hogHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(hogHist)

        rgbHist = self.__xRGBHist(image_src)
        rgbHist = np.asarray(rgbHist)
        cv2.normalize(rgbHist, rgbHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(rgbHist)

        fv = np.reshape(fv, -1)
        
        for i in fv:
            fp.write("%s " % i)
        fp.write("\n")

    def __xRGBHist(self, image):
        numBins = 32
    
        bCh, gCh, rCh = cv2.split(image)

        bins = np.arange(numBins).reshape(numBins,1)
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
        
        rgbHist = []
        for item,col in zip([bCh, gCh, rCh],color):
            hist_item = cv2.calcHist([item],[0],None,[numBins],[0,255])
            rgbHist.extend(hist_item)

        return rgbHist
    
    def __xHOGHist(self, image):
    
        nBins = 72

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageBlur = cv2.GaussianBlur(image, (5,5), 0)
        
        fdescriptor = hog(imageBlur, orientations=nBins, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), visualise=False)
        
        idx = 0
        count = 1
        fHist = np.zeros(nBins)
        for val in fdescriptor:
            fHist[idx] += val
            count += 1
            idx += 1
            if count%nBins == 1:
                idx = 0

        return fHist
    
    def __xSurfHist(self, image):
    
        nBins = 64
        hessian_threshold = 500
        nOctaves = 4
        nOctaveLayers = 2
    
        imageGS = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        surf = cv2.SURF(hessian_threshold, nOctaves, nOctaveLayers, False, True)
        keypoints, descriptors = surf.detectAndCompute(imageGS, None) 
        
        surfHist = np.zeros(nBins)

        if len(keypoints) > 0:
    
            surfHist = np.zeros(nBins)
            for val in descriptors:
                idx = 0
                rowFeatures = np.array(val, dtype = np.float32)
                for val2 in rowFeatures:
                    surfHist[idx] += val2
                    idx += 1
    
        return surfHist

    def __set_timer(self, mesg=""):
        self.timer = time.time()
        if len(mesg) > 0:
            print "Starting ", mesg, "..."

    def __print_timer(self, mesg=""):
        print mesg, "done. run time = ", time.time() - self.timer

    def dumpFeatureVector(self, fp):
        # new_name = string.replace(self.image_src, "jpg", "fv")

        # fp = open(file_name, 'w')
        fv = []

        color_norm = np.linalg.norm((256, 256, 256))
        
        #bg_r, bg_g, bg_b = self.most_occured_color
        #fg_r, fg_g, fg_b = self.ave_color[self.saliency_list[0][1]]
        #print self.most_occured_color
        #print color_norm
        for i in xrange(NUM_OF_SALIENT_OBJECTS):
            segment_id = self.saliency_list[i][1]
            x_pos, y_pos = self.positions[segment_id]
            saliency = self.saliency_list[i][0]
            size = self.pixel_count[segment_id]
            color = self.ave_color[segment_id]
            dist1 = np.linalg.norm(self.most_occured_color - color)/color_norm
            dist2 = np.linalg.norm(self.ave_color[self.saliency_list[0][1]] - color)/color_norm
            fv.append(x_pos)
            fv.append(y_pos)
            fv.append(saliency)
            fv.append(size)
            fv.append(dist1)
            fv.append(dist2)
        
        fHOG = self.__computeHOG(self.image_src)
        #print fHOG.shape
        #print len(fv)
        fv = np.reshape(fv, -1)
        fv = np.concatenate([fv, fHOG], axis=0)
        #print len(fv)

        fp.write("%s" % self.a_score)
        
        count = 1
        for i in fv:
            fp.write(" %s:%s" % (count, i))
            count += 1

        fp.write('\n')

        # fp.close()
        
    def __remove_non_salient_objects(self, salient_objects, seg_map):
        #seg_map2 = np.empty_like(seg_map)
        seg_map2 = np.copy(seg_map)
        height, width = seg_map.shape
        for i in xrange(height):
            for j in xrange(width):
                if salient_objects[seg_map[i][j]] == 0:
                    #seg_map2[i][j] = -1
                    seg_map2[i][j] = 0

        return seg_map2

    def __find_saliency_of_segments(self, seg_map, sal_map):
        height, width = seg_map.shape
        num_segments = np.amax(seg_map)+1
        #print num_segments
        saliency_list = np.zeros(shape=(num_segments,2), dtype=(float, int))

        for i in xrange(num_segments):
            saliency_list[i][1] = int(i)

        pixel_count = np.zeros(num_segments)
        for i in xrange(height):
            for j in xrange(width):
                if seg_map[i][j] != self.deleted_segment:
                    saliency_list[seg_map[i][j]][0] += sal_map[i][j]
                    pixel_count[seg_map[i][j]] += 1

        #print sorted(pixel_count)
        #print pixel_count
        
        for i in xrange(num_segments):
            #print pixel_count[i]
            #print saliency_list[i][0]
            #print saliency_list[i][1]

            saliency_list[i][0] = saliency_list[i][0]/(pixel_count[i]+1)

        #print saliency_list
        saliency_list = sorted(saliency_list, key=lambda x: x[0], reverse=True)
        #saliency_list.sort()
        #print saliency_list

        salient_objects = np.zeros(num_segments)
        for i in xrange(NUM_OF_SALIENT_OBJECTS):
            #print saliency_list[i][0]
            #print saliency_list[i][1]
            salient_objects[int(saliency_list[i][1])] = 1
        
        return saliency_list, salient_objects, pixel_count

    def __find_features(self, seg_map, pixel_count, saliency_list, salient_objects):

        height, width, dim = self.image.shape
        num_segments = np.amax(seg_map)+1
        cumulative_position = np.zeros(shape=(num_segments, 2))
        ave_color = np.zeros(shape=(num_segments, dim))

        #color_occurence = np.zeros(shape=(256, 256, 256))
        
        #print np.amax(pixel_count)
        #max_color_seg = np.argmax(pixel_count)
        salient_objects[self.prominent_color_segment] = 1

        for i in xrange(height):
            for j in xrange(width):
                if salient_objects[seg_map[i][j]] == 1:
                    cumulative_position[seg_map[i][j]] += i, j
                    ave_color[seg_map[i][j]] += self.image[i][j]

                #r, g, b = self.image[i][j]
                #color_occurence[r][g][b] += 1

        #self.max_color = np.amax(color_occurence)
        #print self.max_color
        # self.rmax, self.gmax, self.bmax = np.amax(np.asarray(color_occurence))

        for i in xrange(num_segments):
            if salient_objects[i] == 1:
                cumulative_position[i] = cumulative_position[i]/(pixel_count[i]+1)
                cumulative_position[i][0] = cumulative_position[i][0]/height
                cumulative_position[i][1] = cumulative_position[i][0]/width                

                ave_color[i] = ave_color[i]/(pixel_count[i]+1)
                pixel_count[i] = pixel_count[i]/(height*width)

        self.most_occured_color = ave_color[self.prominent_color_segment]

        #print ave_color
        #print cumulative_position
        return ave_color, cumulative_position

    def __computeHOG(self, image_src, nBins = 72):
        image = cv2.imread(image_src, 0)
        image_blur = cv2.GaussianBlur(image, (5,5), 0)
        fd = hog(image_blur, orientations=nBins, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualise=False)

        idx = 0
        count = 1
        fHist = np.zeros(nBins)
        for val in fd:
            fHist[idx] += val
            count += 1
            idx += 1
            if count%nBins == 1:
                idx = 0

        featureHist = np.asarray(fHist)
        cv2.normalize(featureHist, featureHist, 0, 1, cv2.NORM_MINMAX)
        
        return featureHist

    def __plot_segment(self, seg_number):
        height, width = self.segment_map.shape
        for i in xrange(height):
            for j in xrange(width):
                if self.segment_map[i][j] != seg_number:
                    self.segment_map[i][j] = 0


    def __plot_maps(self, db_path):

        #self.__plot_segment(125)

        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(8, 3, forward=True)
        plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
        
        ax[0].imshow(self.image)
        #ax[0].imshow(self.slic_map)
        ax[0].set_title("Image")
        #ax[1].imshow(mark_boundaries(self.image, self.slic_map))
        ax[1].imshow(self.saliency_map, interpolation='nearest')
        ax[1].set_title("Saliency")
        ax[2].imshow(mark_boundaries(self.image, self.segment_map))
        #ax[2].imshow(mark_boundaries(self.image, self.dummy_map))
        ax[2].set_title("SLIC")
        ax[3].imshow(self.segment_map2, interpolation='nearest')
        #ax[3].imshow(self.segment_map)
        ax[3].set_title("Segmented Image")
        file_name = db_path + "/salientObjects.png"
        plt.savefig(file_name, dpi=500)
