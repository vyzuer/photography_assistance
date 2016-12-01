from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.segmentation import relabel_sequential
from skimage import img_as_float
from skimage.color import rgb2gray, rgb2hsv
from matplotlib import pyplot as plt
import numpy as np
import string
from scipy import ndimage
import os
import time
import cv2
from skimage.feature import hog
from skimage import morphology

import segmentation as seg
import Saliency as saliency
import FaceDetection as face_detection
from sklearn.externals import joblib
from skimage import transform as tf

import cython_utils as cutils

from multiprocessing import Process, Array
from numpy.core.umath_tests import inner1d

NUM_OF_SALIENT_OBJECTS = 30
FV_LENGTH = 200

x_scale = 4
y_scale = 3

class SalientObjectDetection:
    def __init__(self, image_src, segment_dump="/tmp/", a_score=1, prediction_stage=False):
        self.timer = time.time()
        self.image_src = image_src
        self.a_score = a_score
        self.image = io.imread(image_src)
        print "Image source : ", image_src

        self.__set_timer("segmentation...")
        segment_object = seg.SuperPixelSegmentation(self.image)
        
        self.segment_map = segment_object.getSegmentationMap()
        #np.savetxt("segment_map.txt", self.segment_map, '%d')
        # self.slic_map = segment_object.getSlicSegmentationMap()
        #self.dummy_map = segment_object.getDummyMap()
        # self.prominent_color_segment = segment_object.getProminentColorSegment()
        # self.deleted_segment = segment_object.getDeletedSegmentID()
        # self.mean_color, self.fwd, self.inv  = segment_object.getMeanColor()
        self.__print_timer("segmentation")

        self.__set_timer("saliency...")
        if prediction_stage==False:
            saliency_object = saliency.Saliency(self.image, 3)
            self.saliency_map = saliency_object.getSaliencyMap()
        else:
            # self.saliency_map = self.__predict_saliency_map()
            saliency_object = saliency.Saliency(self.image, 3)
            self.saliency_map = saliency_object.getSaliencyMap()
            # self.segment_map2 = self.segment_map

        self.__print_timer("saliency")

        # perform face detection
        self.__set_timer("face detection...")
        self.faces = face_detection.detect(self.image)
        print len(self.faces)
        self.__print_timer("face detection")

        self.__set_timer("saliency detection of objects...")
        # modify saliency using detected faces
        # only for visualization purpose
        # self.saliency_map2 = self.__modify_saliency(self.saliency_map, self.faces)
        if prediction_stage == False:
            # self.saliency_list, self.salient_objects, self.pixel_count = self.__find_saliency_of_segments_2(self.segment_map, self.saliency_map)
            self.saliency_list, self.salient_objects, self.pixel_count, self.segment_map2 = cutils.detect_saliency_of_segments(self.segment_map.astype(np.intp), self.saliency_map)
        else:
            self.saliency_list2, self.salient_objects2, self.pixel_count2, self.segment_map2 = cutils.detect_saliency_of_segments(self.segment_map.astype(np.intp), self.saliency_map)
            self.saliency_list, self.salient_objects, self.pixel_count = self.__predict_saliency_of_segments(segment_dump)
        self.__print_timer("saliency detection of objects")
        # self.ave_color, self.positions = self.__find_features(self.segment_map, self.pixel_count, self.saliency_list, self.salient_objects)

        # self.__set_timer("removing non-salienct objects...")
        # self.segment_map2 = self.__remove_non_salient_objects(self.salient_objects, self.segment_map)
        # self.__print_timer("removing non-salient objects")
    
        self.__set_timer("composition...")

        # modify segment map for faces
        self.segment_map2 = self.__modify_segment_map_for_faces()

        full_comp = True
        if prediction_stage == False:
            self.composition, self.feature_vector = self.__find_composition(grid_size = (3, 4), block_size = (8, 8), full_composition=full_comp)
        else:
            self.composition = None
            self.feature_vector = None

        self.new_feature_vector = self.__find_composition_vector(full_composition=full_comp)

        self.face_vector = self.__find_face_features()

        self.view_context = self.__find_view_context()

        self.__print_timer("composition")


    def __find_face_features(self):
        img_h, img_w = self.segment_map.shape
        img_size = img_h*img_w

        num_faces = len(self.faces)
        max_size = 0.0

        descriptor = np.zeros(2)

        for (x, y, w, h) in self.faces:
            # x - horizontal position
            # y - vertical position
            # w - width
            # h - height
            size = w*h
            if(size > max_size):
                max_size = size

        descriptor[0] = num_faces
        descriptor[1] = 1.0*max_size/img_size
        # print descriptor
        return descriptor

    def __find_view_context(self, grid_size = (4, 4), block_size = (2, 2), nbins = 32):
        """
        Extract photographic view-context for an image

        """

        n_xcells, n_ycells = grid_size
        n_bxcells, n_bycells = block_size

        # cell size for a image
        img_height, img_width = self.segment_map.shape
        x_step = 1.0*img_height/n_xcells
        y_step = 1.0*img_width/n_ycells

        # block steps
        bx_step = x_step*n_bxcells
        by_step = y_step*n_bycells

        n_xblocks = n_xcells - n_bxcells + 1
        n_yblocks = n_ycells - n_bycells + 1

        n_dims = 3*nbins*n_xblocks*n_yblocks
        feature_vector = np.zeros(n_dims)

        bOptimize = True

        if bOptimize == True:
            feature_vector = cutils.find_view_context(self.image, n_xcells, n_ycells, n_bxcells, n_bycells, nbins)
        else:

            # img_hsv = rgb2hsv(self.image)

            x_pos = 0.0
            y_pos = 0.0
            count = 0
            for i in range(n_xblocks):
                # reset y_pos to 0
                y_pos = 0.0
                for j in range(n_yblocks):
                    x_min = int(x_pos)
                    y_min = int(y_pos)
                    x_max = int(x_pos + bx_step)
                    y_max = int(y_pos + by_step)

                    # extract color histogram for this block
                    # img_block = img_hsv[x_min:x_max, y_min:y_max,:]
                    img_block = self.image[x_min:x_max, y_min:y_max,:]
                    # print img_block.shape

                    h, s, v = cv2.split(img_block)

                    hist1, bin_edges1 = np.histogram(h, nbins, (0,1))
                    hist2, bin_edges2 = np.histogram(s, nbins, (0,1))
                    hist3, bin_edges3 = np.histogram(v, nbins, (0,1))

                    feature_vector[count:count+nbins] = hist1
                    count += nbins
                    feature_vector[count:count+nbins] = hist2
                    count += nbins
                    feature_vector[count:count+nbins] = hist3
                    count += nbins

                    # increment by one cell step
                    y_pos = (j+1)*y_step

                # increment x_step by one cell
                x_pos = (i+1)*x_step

            cv2.normalize(feature_vector, feature_vector, 0, 1, cv2.NORM_MINMAX)

        return feature_vector



    def __find_internal_saliency_map(self, seg_map):

        img_height, img_width = seg_map.shape

        saliency_list = self.saliency_list

        img_height, img_width = seg_map.shape

        new_saliency_map = np.zeros(shape=seg_map.shape)
        for i in range(img_height):
            for j in range(img_width):
                seg_id = seg_map[i][j]
                new_saliency_map[i][j] = saliency_list[seg_id]

        return new_saliency_map
        
    def getNewSaliencyMap(self):
        return self.new_saliency_map
        
    def getFaceInfo(self):
        return self.faces

    def get_photograph_composition(self):
        return self.composition, self.new_feature_vector, self.face_vector, self.view_context


    def get_photograph_composition_full(self):

        self.feature_map_full, self.feature_vector_full = self.__find_composition(full_composition=True)

        return self.feature_map_full, self.feature_vector_full


    def get_photograph_composition_avg(self):

        self.feature_map_avg, self.feature_vector_avg = self.__find_composition(average=True)

        return self.feature_map_avg, self.feature_vector_avg


    def get_photograph_composition_full_avg(self):

        self.feature_map_full_avg, self.feature_vector_full_avg = self.__find_composition(full_composition=True, average=True)

        return self.feature_map_full_avg, self.feature_vector_full_avg


    def __modify_saliency(self, saliency_map, faces):
        # traverse through all the faces found
        saliency_map2 = np.copy(saliency_map)
        for (x, y, w, h) in faces:
            # x - horizontal position
            # y - vertical position
            # w - widht
            # h - height
            for i in range(w):
                for j in range(h):
                    # saliency_map[y+j][x+i] = 1.0
                    mod_sal = 0.5*(2 - np.absolute(0.5 - 1.0*i/w) - np.absolute(0.5 - 1.0*j/h))
                    saliency_map2[y+j][x+i] = mod_sal
                    # print i, j, w, h, mod_sal

        return saliency_map2


    def __modify_segment_map_for_faces(self):
        seg_map = self.segment_map2

        faces = self.faces
        # total segments, +1 for faces
        num_segments = len(self.saliency_list)

        img_height, img_width = seg_map.shape

        # add face segment to segment map
        face_seg_id = num_segments - 1

        # traverse through all the faces found
        for (x, y, w, h) in faces:
            # x - horizontal position
            # y - vertical position
            # w - width
            # h - height
            for i in range(w):
                for j in range(h):
                    seg_map[y+j][x+i] = face_seg_id
                    self.segment_map[y+j][x+i] = face_seg_id

        return seg_map


    def __find_composition_vector(self, grid_size = (9, 9), block_size = (3, 3), nbins = 20, full_composition=False):
        """
        Extract photographic composition vector for an image

        """
        feature_vector = []
        n_xcells, n_ycells = grid_size
        n_bxcells, n_bycells = block_size

        if full_composition == False:
            seg_map = self.segment_map2
        else:
            seg_map = self.segment_map

        bOptimize = True
        # bOptimize = False

        if bOptimize == True:
            feature_vector, self.new_saliency_map = cutils.find_comp_vec(seg_map, self.saliency_list, n_xcells, n_ycells, n_bxcells, n_bycells, nbins)
        else:
            self.new_saliency_map = self.__find_internal_saliency_map(seg_map)

            # cell size for a image
            img_height, img_width = self.segment_map.shape
            x_step = 1.0*img_height/n_xcells
            y_step = 1.0*img_width/n_ycells

            # block steps
            bx_step = x_step*n_bxcells
            by_step = y_step*n_bycells

            n_xblocks = n_xcells - n_bxcells + 1
            n_yblocks = n_ycells - n_bycells + 1

            n_dims = nbins*n_xblocks*n_yblocks
            feature_vector = np.zeros(n_dims)

            x_pos = 0.0
            y_pos = 0.0
            count = 0
            for i in range(n_xblocks):
                # reset y_pos to 0
                y_pos = 0.0
                for j in range(n_yblocks):
                    x_min = int(x_pos)
                    y_min = int(y_pos)
                    x_max = int(x_pos + bx_step)
                    y_max = int(y_pos + by_step)

                    # extract histogram for this block
                    saliency_block = self.new_saliency_map[x_min:x_max, y_min:y_max]
                    hist, bin_edges = np.histogram(saliency_block, nbins, (0,1))
                    print hist

                    feature_vector[count:count+nbins] = hist
                    count += nbins

                    # increment by one cell step
                    y_pos = (j+1)*y_step

                # increment x_step by one cell
                x_pos = (i+1)*x_step

        # print feature_vector.shape
        cv2.normalize(feature_vector, feature_vector, 0, 1, cv2.NORM_MINMAX)
        return feature_vector


    def __find_composition(self, grid_size = (3, 4), block_size = (8, 8), full_composition=False, average=False):
        """
        Extract photographic composition for an image

        """

        grid_height, grid_width = grid_size
        block_height, block_width = block_size

        f_map_height = grid_height*block_height
        f_map_width = grid_width*block_width

        feature_map = np.zeros(shape=(f_map_height, f_map_width))
        feature_vector = np.zeros(f_map_height*f_map_width)
        
        # number of segments, some of them might not be present
        if full_composition == False:
            seg_map = self.segment_map2
        else:
            seg_map = self.segment_map

        img_height, img_width = seg_map.shape

        # cell size for a image
        x_step = 1.0*img_height/f_map_height
        y_step = 1.0*img_width/f_map_width
        t_step = x_step*y_step

        bOptimize= True

        if bOptimize == True:
            feature_map = cutils.comp_map(grid_height, grid_width, block_height, block_width, seg_map, self.saliency_list)
        else:
            # block steps
            bx_step = 1.0*img_height/grid_height
            by_step = 1.0*img_width/grid_width

            # current location of our spanning
            pix_x = 0.0
            pix_y = 0.0
            block_x = 0.0
            block_y = 0.0
            cnt = 0
            for i in range(grid_height):
                pix_x = block_x
                block_y = 0.0
                for j in range(grid_width):
                    pix_x = block_x
                    pix_y = block_y
                    for k in range(block_height):
                        x_0 = pix_x
                        pix_x += x_step

                        pix_y = block_y
                        for l in range(block_width):
                            y_0 = pix_y
                            pix_y += y_step
                            saliency_sum = 0.0
                            for m in range(int(x_0), int(pix_x)):
                                for n in range(int(y_0), int(pix_y)):
                                    # print i, j, k, l, m, n
                                    seg_id = seg_map[m][n]
                                    saliency_sum += self.saliency_list[seg_id]

                            saliency_norm = saliency_sum/t_step
                            # saliency_norm = saliency_sum/((pix_x - x_0)*(pix_y - y_0))
                            # print saliency_norm
                            ii = i*block_height + k
                            jj = j*block_width + l
                            feature_map[ii][jj] = saliency_norm
                            feature_vector[cnt] = saliency_norm
                            cnt +=1
                    
                    block_y += by_step
                
                block_x += bx_step


            # find average saliency using block size
            feature_map_2 = None
            if average == True :
                feature_map_2 = np.copy(feature_map)
                n_cellsx, n_cellsy = np.shape(feature_map)
                
                # block size for average calculation
                b_sizex, b_sizey = 3, 3
                t_cells = b_sizex*b_sizey

                # each block will span 3 cells are a time
                bx = n_cellsx - b_sizex + 1
                by = n_cellsy - b_sizey + 1

                for i in range(bx):
                    for j in range(by):
                        s_t = np.sum(feature_map[i:i+b_sizex, j:j+b_sizey])
                        feature_map_2[i+1, j+1] = s_t/t_cells
                        
                pix_x = 0.0
                pix_y = 0.0
                block_x = 0.0
                block_y = 0.0
                cnt = 0
                for i in range(grid_height):
                    pix_x = block_x
                    block_y = 0.0
                    for j in range(grid_width):
                        pix_x = block_x
                        pix_y = block_y
                        for k in range(block_height):
                            x_0 = pix_x
                            pix_x += x_step

                            pix_y = block_y
                            for l in range(block_width):
                                y_0 = pix_y
                                pix_y += y_step
                                ii = i*block_height + k
                                jj = j*block_width + l
                                feature_vector[cnt] = feature_map_2[ii][jj]
                                cnt +=1
                        
                        block_y += by_step
                    
                    block_x += bx_step


        if average == True :
            return feature_map_2, feature_vector
        else:
            return feature_map, feature_vector


    def __predict_saliency_of_segments(self, seg_dir):
        # find all the segments and classify them as one of the known objects 
        # to find out the saliency value
        img_x, img_y = self.segment_map2.shape
        segs = ndimage.find_objects(self.segment_map)
        # print self.segment_map
        # ignore the last segment as it is for faces
        num_segments = np.amax(self.segment_map)+1
        # num_segments = len(segs)
        # print num_segments
        saliency_list = np.zeros(num_segments+1)
        pixel_count = np.zeros(num_segments)

        # THese will be used for affine transforamtion and camera motion prediction
        actual_position = np.zeros(shape=(num_segments, 2))
        target_position = np.zeros(shape=(num_segments, 2))

        # load the cluster model for classification and popularity score
        cluster_dump = seg_dir + "/cluster_model/cluster.pkl"
        cluster_model = joblib.load(cluster_dump)

        popularity_dump = seg_dir + "/popularity.score"
        popularity_score = np.loadtxt(popularity_dump)

        # max_dist = cluster_model.inertia_/len(cluster_model.labels_)
        # print max_dist
        # print cluster_model.inertia_
        # print len(cluster_model.labels_)
        for i in xrange(num_segments-1):
            if segs[i] == None:
                continue

            segment_img = self.image[segs[i]]

            # find the position
            x_0 =  segs[i][0].start
            x_1 =  segs[i][0].stop
            y_0 = segs[i][1].start
            y_1 = segs[i][1].stop
            x_pos = ((x_1 + x_0)/2.0)/img_x
            y_pos = ((y_1 + y_0)/2.0)/img_y

            segment_copy = np.copy(segment_img)

            mask = self.segment_map[segs[i]]
            idx=(mask!=i+1)
            segment_copy[idx] = 255, 255, 255
            
            fv = self.__find_segment_features(segment_img, segment_copy)
            
            # classify segment
            segment_id = cluster_model.predict(fv)[0]
            p_score = popularity_score[segment_id]

            # print p_score
            # print segment_id
            # dist = inner1d(fv, cluster_model.cluster_centers_[segment_id])
            # print i
            # print dist

            # saliency_list[i+1] = p_score
            saliency_list[i+1] = (p_score+self.saliency_list2[i+1])/2

            # find the target position for this segment
            gmm_path = seg_dir + "/SegClustersInfo/" + str(segment_id) + "/gmm_dumps/gmm.pkl"

            if os.path.isfile(gmm_path):
                gmm_model = joblib.load(gmm_path)

                x_ = y_scale - y_scale*x_pos
                y_ = x_scale*y_pos

                c_pos = np.array([x_, y_])
                gmm_cluster = gmm_model.predict([c_pos])
                # print gmm_cluster
                # print gmm_model.means_
                x_pred, y_pred = gmm_model.means_[gmm_cluster[0]]

                target_position[i+1] = x_pred, y_pred
                actual_position[i+1] = x_, y_

                # print x_pred, y_pred
            else:
                saliency_list[i+1] = 0.0

        saliency_list = saliency_list/saliency_list.max()
        # face saliency
        saliency_list[num_segments] = 1.0

        salient_objects = np.ones(num_segments)
        actual_p = []
        target_p = []
        for i in xrange(num_segments):
            if saliency_list[i] < 0.10 :
                salient_objects[i] = 0
            else:
                actual_p.append(actual_position[i])
                target_p.append(target_position[i])

        self.actual_p = np.array(actual_p)
        self.target_p = np.array(target_p)

        # print self.actual_p.shape

        return saliency_list, salient_objects, pixel_count


    def estimate_affine_parameters(self):
        t_form = tf.estimate_transform('affine', self.actual_p, self.target_p)

        tilt = t_form.translation[0]/y_scale
        pan = t_form.translation[1]/x_scale
        zoom = np.mean(t_form.scale)
        rotate = t_form.rotation*180*7/22

        print 'Pan:{0:.4f} Tilt:{1:.4f} Zoom:{2:.4f}'.format(pan, tilt, zoom)

        return t_form, tilt, pan, zoom, rotate


    def process_segments(self, dump_path):
        img_x, img_y = self.segment_map2.shape
        segs = ndimage.find_objects(self.segment_map2)
        # ignore the last segment as it is for faces
        num_segments = len(segs) - 1

        # 0 is for backgroud
        # last segment is for faces
        # num_segments = len(self.saliency_list) - 1

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
        pos_file = dir_path + "/pos.list"

        if os.path.isfile(feature_file):
            os.unlink(feature_file)
        if os.path.isfile(saliency_file):
            os.unlink(saliency_file)
        if os.path.isfile(pos_file):
            os.unlink(pos_file)

        fp = open(saliency_file, 'w')
        fp1 = open(feature_file, 'a')
        fp2 = open(pos_file, 'w')

        # self.plot_maps(dir_path)

        j = 1
        for i in xrange(num_segments):

            # for deleted segments
            if segs[i] == None:
                continue

            fp.write("%0.8f\n" % self.saliency_list[i+1])

            segment_img = self.image[segs[i]]

            # find the position
            x_0 =  segs[i][0].start
            x_1 =  segs[i][0].stop
            y_0 = segs[i][1].start
            y_1 = segs[i][1].stop
            x_pos = (x_1 + x_0)/2.0
            y_pos = (y_1 + y_0)/2.0
            # print segs[i]
            # print '{0} {1}'.format(x_pos, y_pos)

            fp2.write("{0:0.8f} {1:0.8f}\n".format(x_pos/img_x, y_pos/img_y))

            segment_copy = np.copy(segment_img)

            mask = self.segment_map2[segs[i]]
            idx=(mask!=i+1)
            segment_copy[idx] = 255, 255, 255
            
            fig, ax = plt.subplots(1, 2)

            ax[0].axis('off')
            ax[1].axis('off')
            fig.patch.set_visible(False)

            ax[0].imshow(segment_img)
            ax[1].imshow(segment_copy)

            file_name = seg_path + str(j) + ".png"
            plt.savefig(file_name, dpi=60)
            plt.close()
            
            self.__dump_segment_features(segment_img, segment_copy, fp1)

            j += 1

        fp.close()
        fp1.close()
        fp2.close()

    def __find_segment_features(self, image_src, segment_copy):
        fv = []
    
        # self.__set_timer("surf")
        surfHist = self.__xSurfHist(segment_copy)
        surfHist = np.asarray(surfHist)
        cv2.normalize(surfHist, surfHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(surfHist)
        # self.__print_timer("surf")

        # self.__set_timer("hog")
        hogHist = self.__xHOGHist(segment_copy)
        hogHist = np.asarray(hogHist)
        cv2.normalize(hogHist, hogHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(hogHist)
        # self.__print_timer("hog")

        # self.__set_timer("rgb")
        rgbHist = self.__xRGBHist(segment_copy)
        rgbHist = np.asarray(rgbHist)
        cv2.normalize(rgbHist, rgbHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(rgbHist)
        # self.__print_timer("rgb")

        fv = np.reshape(fv, -1)

        return fv


    def __dump_segment_features(self, image_src, segment_copy, fp):

        fv = self.__find_segment_features(image_src, segment_copy)
        np.savetxt(fp, np.atleast_2d(fv), fmt='%.8f')
        
        # for i in fv:
        #     fp.write("%s " % i)
        # fp.write("\n")


    def __xRGBHist(self, image):
        numBins = 256
    
        bCh, gCh, rCh = cv2.split(image)

        bins = np.arange(numBins).reshape(numBins,1)
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
        
        rgbHist = []
        for item,col in zip([bCh, gCh, rCh],color):
            hist_item = cv2.calcHist([item],[0],None,[numBins],[0,255])
            rgbHist.extend(hist_item)

        return rgbHist

    
    def __xHOGHist(self, image):
    
        nBins = 32

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageBlur = cv2.GaussianBlur(image, (5,5), 0)
        
        fdescriptor = hog(imageBlur, orientations=nBins, pixels_per_cell=(16, 16),
                                    cells_per_block=(1, 1), visualise=False)
        

        fd = np.reshape(fdescriptor, (-1, nBins))
        fHist = np.sum(fd, axis=0)

        # idx = 0
        # count = 1
        # fHist = np.zeros(nBins)
        # for val in fdescriptor:
        #     fHist[idx] += val
        #     count += 1
        #     idx += 1
        #     if count%nBins == 1:
        #         idx = 0

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
            surfHist = np.sum(descriptors, axis=0)
    
        #     surfHist = np.zeros(nBins)
        #     for val in descriptors:
        #         idx = 0
        #         rowFeatures = np.array(val, dtype = np.float32)
        #         for val2 in rowFeatures:
        #             surfHist[idx] += val2
        #             idx += 1
    
        return surfHist


    def __set_timer(self, mesg=""):
        self.timer = time.time()
        if len(mesg) > 0:
            print "Starting ", mesg, "..."

    def __print_timer(self, mesg=""):
        print mesg, "done. run time = ", time.time() - self.timer

    def dump_feature_vector(self, fp):
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

    def __find_saliency_of_segments_2(self, seg_map, sal_map):
        height, width = seg_map.shape
        num_segments = np.amax(seg_map)+1
        #print num_segments
        # +1 for face segments
        saliency_list = np.zeros(num_segments+1)

        pixel_count = np.zeros(num_segments)
        for i in xrange(height):
            for j in xrange(width):
                if seg_map[i][j] != 0:
                    seg_id = seg_map[i][j]
                    saliency_list[seg_id] += sal_map[i][j]
                    pixel_count[seg_id] += 1

        #print sorted(pixel_count)
        #print pixel_count
        
        for i in xrange(num_segments):
            saliency_list[i] = saliency_list[i]/(pixel_count[i]+1)

        # saliency_list = saliency_list - saliency_list.min()
        saliency_list = saliency_list/saliency_list.max()
        salient_objects = np.ones(num_segments)
        for i in xrange(num_segments):
            if saliency_list[i] < 0.10 :
                salient_objects[i] = 0
        
        return saliency_list, salient_objects, pixel_count


    def __find_saliency_of_segments(self, seg_map, sal_map):
        height, width = seg_map.shape
        num_segments = np.amax(seg_map)+1
        #print num_segments
        saliency_list = np.zeros(shape=(num_segments,2), dtype=(float, int))

        for i in xrange(num_segments):
            saliency_list[i][1] = i

        pixel_count = np.zeros(num_segments)
        for i in xrange(height):
            for j in xrange(width):
                if seg_map[i][j] != 0:
                    seg_id = int(seg_map[i][j])
                    saliency_list[seg_id][0] += sal_map[i][j]
                    pixel_count[seg_id] += 1

        #print sorted(pixel_count)
        #print pixel_count
        
        # for i in xrange(num_segments):
        #     #print pixel_count[i]
        #     #print saliency_list[i][0]
        #     #print saliency_list[i][1]

        #     saliency_list[i][0] = saliency_list[i][0]/(pixel_count[i]+1)

        #print saliency_list
        saliency_list = sorted(saliency_list, key=lambda x: x[0], reverse=True)
        #saliency_list.sort()
        #print saliency_list

        salient_objects = np.zeros(num_segments)
        for i in xrange(num_segments):
            #print saliency_list[i][0]
            #print saliency_list[i][1]
            salient_objects[int(saliency_list[i][1])] = 1
        
        return saliency_list, salient_objects, pixel_count

    def __find_features(self, seg_map, pixel_count, saliency_list, salient_objects):

        height, width, dim = self.image.shape
        # segment for face is not required here
        num_segments = len(saliency_list) - 1
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


    def plot_composition_map(self, db_path):

        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(8, 3, forward=True)
        plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
        
        # fig.patch.set_visible(False)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')

        ax[0].imshow(self.composition)
        # ax[0].imshow(self.slic_map)
        ax[0].set_title("composition - 1")

        # ax[1].imshow(mark_boundaries(self.image, self.slic_map))
        ax[1].imshow(self.feature_map_full, interpolation='nearest')
        ax[1].set_title("composition - 2")

        ax[2].imshow(self.feature_map_avg, interpolation='nearest')
        #ax[3].imshow(self.segment_map)
        ax[2].set_title("composition average")

        ax[3].imshow(self.new_saliency_map, interpolation='nearest')
        # ax[3].imshow(self.segment_map, interpolation='nearest')
        #ax[2].imshow(mark_boundaries(self.image, self.dummy_map))
        ax[3].set_title("composition avg - 2")

        db, img_name = os.path.split(self.image_src)
        file_name = db_path + "/comp_" + img_name
        plt.savefig(file_name, dpi=300)
        plt.close()

    def plot_maps_2(self, db_path):

        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(8, 3, forward=True)
        plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
        
        # fig.patch.set_visible(False)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')

        ax[0].imshow(self.image)
        # ax[0].imshow(self.slic_map)
        ax[0].set_title("Image")

        # ax[1].imshow(mark_boundaries(self.image, self.slic_map))
        # ax[1].imshow(self.saliency_map, interpolation='nearest')
        ax[1].imshow(self.segment_map2, interpolation='nearest')
        ax[1].set_title("Saliency")

        # ax[2].imshow(self.segment_map2, interpolation='nearest')
        ax[2].imshow(self.new_saliency_map, interpolation='nearest')
        #ax[3].imshow(self.segment_map)
        ax[2].set_title("Segmented Image")

        # ax[3].imshow(self.composition, interpolation='nearest')
        # ax[3].imshow(self.new_saliency_map, interpolation='nearest')
        ax[3].imshow(self.segment_map, interpolation='nearest')
        #ax[2].imshow(mark_boundaries(self.image, self.dummy_map))
        ax[3].set_title("Composition")

        db, img_name = os.path.split(self.image_src)
        file_name = db_path + "/" + img_name
        plt.savefig(file_name, dpi=100)
        plt.close()

    def plot_maps(self, db_path):

        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(8, 3, forward=True)
        plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
        
        # fig.patch.set_visible(False)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')

        ax[0].imshow(self.image)
        # ax[0].imshow(self.slic_map)
        ax[0].set_title("Image")

        # ax[1].imshow(mark_boundaries(self.image, self.slic_map))
        ax[1].imshow(self.saliency_map, interpolation='nearest')
        ax[1].set_title("Saliency")

        # ax[2].imshow(self.segment_map2, interpolation='nearest')
        ax[2].imshow(self.new_saliency_map, interpolation='nearest')
        #ax[3].imshow(self.segment_map)
        ax[2].set_title("Segmented Image")

        ax[3].imshow(self.composition, interpolation='nearest')
        # ax[3].imshow(self.new_saliency_map, interpolation='nearest')
        # ax[3].imshow(self.segment_map, interpolation='nearest')
        #ax[2].imshow(mark_boundaries(self.image, self.dummy_map))
        ax[3].set_title("Composition")

        db, img_name = os.path.split(self.image_src)
        file_name = db_path + "/" + img_name
        plt.savefig(file_name, dpi=100)
        plt.close()


    def __plot_maps(self, db_path):

        #self.__plot_segment(125)

        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(8, 3, forward=True)
        plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
        
        ax[0].imshow(self.image, interpolation='nearest')
        # ax[0].imshow(self.slic_map)
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
        plt.close()

