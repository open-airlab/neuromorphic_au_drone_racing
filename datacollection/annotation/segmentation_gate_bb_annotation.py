import cv2
import os
import json
import shutil
import numpy as np
from shutil import copyfile
import json
import math

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir) 
from mydataloader.prophesee import npy_events_tools

from quaternion_utils import Quaternion



class GateBoundingBoxAnnotation:
    """ Annotate gates for training using back projection

        This class is a converter to 3D coordinates of gates to 2D pixel coordinates.
    """

    def __init__(self):

        self.resolution = [240, 180]
        self.trackid = 0

    def computeBoundingBox(self, seg_image):

        debug_img = None  
        num = 0

        #seg_image = seg_image-130
        seg_image = abs(seg_image - 255)
        
        seg_image = seg_image.astype(np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15)) 
        dilated = cv2.dilate(seg_image, element, iterations=1)
        eroded = cv2.dilate(dilated, element, iterations=1)
        
        # Fill in holes that may remain
        contour,hier = cv2.findContours(eroded,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(eroded,[cnt],0,255,-1)
        
        # blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 25
        params.filterByColor = False
        params.blobColor = 0
        params.filterByArea = False
        params.minArea = 500
        params.maxArea = 5000
        params.filterByCircularity = False
        params.minCircularity =.4
        params.maxCircularity = 1

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)
        
        detector.empty() # <- now works 
        keypoints = detector.detect(eroded)
        
        
        results = []
        
        for keypoint in keypoints:
            
            middle_x = int(keypoint.pt[0])
            middle_y = int(keypoint.pt[1])
            diameter = int(keypoint.size)
            radius = int(diameter/2)
                  
            start_x, start_y = -1, -1
            end_x, end_y = -1, -1
            
            y_start_search = middle_y - radius+10 if middle_y - radius+10 > 0 else 0
            y_end_search = middle_y + radius+10 if middle_y + radius+10 < self.resolution[1] else self.resolution[1]
            x_start_search = middle_x - radius+10 if middle_x - radius+10 > 0 else 0
            x_end_search = middle_x + radius+10 if middle_x + radius+10 < self.resolution[0] else self.resolution[0]
            for y in range(y_start_search, y_end_search):
                if np.sum(seg_image[y, x_start_search:x_end_search]) > 0.0:
                    end_y = y
    
                    if start_y == -1:
                        start_y = y
    
            for x in range(x_start_search, x_end_search):
                if np.sum(seg_image[y_start_search:y_end_search, x]) > 0.0:
                    end_x = x
    
                    if start_x == -1:
                        start_x = x
            if x > 4:
                x = start_x -4
            else:
                x = start_x
            if y > 4:
                y = start_y -4
            else:
                y = start_y
            if end_x + 8 < self.resolution[0]:
                w = end_x - start_x + 8
            else:
                w = end_x - start_x
            if end_y + 8 < self.resolution[1]:
                h = end_y - start_y + 8
            else:
                h = end_y - start_y
            class_id = 0
            confidence = 1
            
            try:
                if h/w < 2:
                    if x >= 0:         
                        results.append((x, y, w, h, class_id, confidence, self.trackid))
                        self.trackid += 1
                        num += 1
                        debug_img = cv2.rectangle(seg_image, (x,y), (x+w, y+h), (255, 0, 0), 2)
            except ZeroDivisionError:
                pass
        
        
        im_with_keypoints = cv2.drawKeypoints(eroded, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)
        
        if debug_img is not None:
            cv2.imshow("segimage", debug_img)
        cv2.waitKey(10)
        lol = 2

        return results, num


    def run(self, source_folder, dest_folder, lookup=None):

        print("[*] Source folder: %s" % source_folder)
        output_dtype = np.dtype([('ts', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')])
        
        # Get all attribute files
        segFiles = []
        for file in os.listdir(source_folder):
            if file.endswith(".npy"):
                segFiles.append(file)
                  

        num_of_samples = len(segFiles)
        print("[*] Num of samples found: %d" % num_of_samples)

        # Create output folder.
        if not os.path.isdir(dest_folder):
            print("%s created." % dest_folder)
            os.mkdir(dest_folder)

        for file in segFiles:
            track_id = 0
            output = np.empty((0,), dtype = output_dtype)
            
            seg_images = np.load(os.path.join(source_folder, file), allow_pickle=True)
            for i, ts in enumerate(seg_images[:,0]):
                            
                results, num = self.computeBoundingBox(seg_images[i, 1])
                
                if num > 0:
                    if num == 1:
                        list_res = list(results[0])
                        list_res.insert(0, ts)
                        res = tuple(list_res)
        
                        temp_arr = np.array(res, dtype=output_dtype)
                        output = np.append(output, temp_arr)
                        
                    elif num > 1:
                        for i in range(num):
                            list_res = list(results[i])
                            list_res.insert(0, ts)
                            res = tuple(list_res)
            
                            temp_arr = np.array(res, dtype=output_dtype)
                            output = np.append(output, temp_arr)
                            
                        
            split_name = file.split('_')
            save_file = split_name[0] + '_' + split_name[1] + '_' + split_name[2] + '_' + split_name[3] + '_bbox.npy'    
            np.save(os.path.join(dest_folder, save_file), output)
                    


def main():
    source_folder = os.path.join(parentdir, 'source_data')
    dest_folder = os.path.join(parentdir, 'dest_data')
    
    bb = GateBoundingBoxAnnotation()
    bb.run(source_folder, dest_folder)
    
if __name__ == "__main__":
    main()