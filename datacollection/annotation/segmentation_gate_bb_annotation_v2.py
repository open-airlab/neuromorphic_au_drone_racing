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
        
        # blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 25
        params.filterByColor = True
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
        keypoints = detector.detect(seg_image)
        
        im_with_keypoints = cv2.drawKeypoints(seg_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)
        
        cv2.imshow("debug_image", seg_image)
        cv2.waitKey(10)
        
        results = []
        
        for keypoint in keypoints:
            
            middle_x = int(keypoint.pt[0])
            middle_y = int(keypoint.pt[1])
            diameter = int(keypoint.size)
            radius = int(diameter/2)
            radius_pos = radius
            radius_neg = radius
            counter = 2
            
            most_left_outer_edge = 240
            most_right_outer_edge = 0
            most_top_outer_edge = 240
            most_bottom_outer_edge = 0
            
            
            if middle_y + radius > len(seg_image)-1:
                radius_pos = len(seg_image)-1 - middle_y
            if middle_y - radius < 0:
                radius_neg = middle_y
            for y in range(-radius_neg, radius_pos):
                left_outer_edge = None
                while left_outer_edge is None:
                    if middle_x - radius - counter < 0:
                        left_outer_edge = 0
                        counter = 2
                    elif seg_image[middle_y + y, middle_x - radius - counter] == 0:
                        # Then we got the first edge
                        left_outer_edge = middle_x - radius - counter
                        counter = 2
                    else:
                        counter += 1
                if left_outer_edge < most_left_outer_edge:
                    most_left_outer_edge = left_outer_edge
                if most_left_outer_edge == 0:
                    break
            radius_neg = radius
            radius_pos = radius
            
            if middle_y + radius > len(seg_image[0])-1:
                radius_pos = len(seg_image[0])-1 - middle_x
            if middle_y - radius < 0:
                radius_neg = middle_x
            for y in range(-radius_neg, radius_pos):
                right_outer_edge = None
                while right_outer_edge is None:
                    if middle_x + radius + counter > len(seg_image[0])-1:
                        right_outer_edge = len(seg_image[0])-1
                        counter = 2
                    elif seg_image[middle_y + y, middle_x + radius + counter] == 0:
                        # Then we got the first edge
                        right_outer_edge = middle_x + radius + counter
                        counter = 2
                    else:
                        counter += 1
                if right_outer_edge > most_right_outer_edge:
                    most_right_outer_edge = right_outer_edge
                if most_right_outer_edge == len(seg_image[0])-1:
                    break
            radius_neg = radius
            radius_pos = radius
                
                
            if middle_x + radius > len(seg_image[0])-1:
                radius_pos = len(seg_image[0])-1 - middle_x
            if middle_x - radius < 0:
                radius_neg = middle_x
            for x in range(-radius_neg, radius_pos):
                top_outer_edge = None
                while top_outer_edge is None:
                    if middle_y - radius - counter < 0:
                        top_outer_edge = 0
                        counter = 2
                    elif seg_image[middle_y - radius - counter, middle_x + x] == 0:
                        # Then we got the first edge
                        top_outer_edge = middle_y - radius - counter
                        counter = 2
                    else:
                        counter += 1
                if top_outer_edge < most_top_outer_edge:
                    most_top_outer_edge = top_outer_edge
                if most_top_outer_edge == 0:
                    break
            radius_neg = radius
            radius_pos = radius
            
            
            if middle_x + radius > len(seg_image[0])-1:
                radius_pos = len(seg_image[0])-1 - middle_x
            if middle_x - radius < 0:
                radius_neg = middle_x
            for x in range(-radius_neg, radius_pos):
                bottom_outer_edge = None
                while bottom_outer_edge is None:
                    if middle_y + radius + counter > len(seg_image)-1:
                        bottom_outer_edge = len(seg_image)-1
                        counter = 2
                    elif seg_image[middle_y + radius + counter, middle_x] == 0:
                        # Then we got the first edge
                        bottom_outer_edge = middle_y + radius + counter
                        counter = 2
                    else:
                        counter += 1
                if bottom_outer_edge > most_bottom_outer_edge:
                    most_bottom_outer_edge = bottom_outer_edge
                if most_bottom_outer_edge == 0:
                    break
            radius_neg = radius
            radius_pos = radius
            
            left_outer_edge = most_left_outer_edge
            right_outer_edge = most_right_outer_edge
            top_outer_edge = most_top_outer_edge            
            bottom_outer_edge = most_bottom_outer_edge
            
            if left_outer_edge > 4:
                x = left_outer_edge - 4
            else:
                x = 0
            if top_outer_edge > 4:
                y = top_outer_edge - 4
            else:
                y = 0
            if right_outer_edge + 8 < len(seg_image[0])-1:
                w = right_outer_edge - x + 4
            else:
                #w = right_outer_edge - left_outer_edge
                w = len(seg_image[0])-1 - x
            if bottom_outer_edge + 8 < len(seg_image)-1:
                h = bottom_outer_edge - y + 4
            else:
                #h = bottom_outer_edge - top_outer_edge
                h = len(seg_image)-1 - y
            
            # Now refine the box
            done = False             
            # Right edge
            while done is False:
                for right_pixel in range(y, y+h):
                    if seg_image[right_pixel, x+w] == 255 and x+w < len(seg_image[0])-1:
                        w += 1
                        break
                    elif right_pixel == y+h -1:
                        done = True
            done = False
            # Left edge
            while done is False:
                for left_pixel in range(y, y+h):
                    if seg_image[left_pixel, x] == 255 and x > 0:
                        x -= 1
                        break
                    elif left_pixel == y+h -1:
                        done = True
            # Top edge
            done = False
            while done is False:
                for top_pixel in range(x, x+w):
                    if seg_image[y, top_pixel] == 255 and y > 0:
                        y -= 1
                        break
                    elif top_pixel == x+w -1:
                        done = True
            done = False
            # Bottom edge
            while done is False:
                for bot_pixel in range(x, x+w):
                    if seg_image[y+h, bot_pixel] == 255 and y+h < len(seg_image)-1:
                        h += 1
                        break
                    elif bot_pixel == x+w -1:
                        done = True
            
            
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
        

        
        if debug_img is not None:
            cv2.imshow("bb_image", debug_img)
        cv2.waitKey(10)

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
        prior_results = None

        # Create output folder.
        if not os.path.isdir(dest_folder):
            print("%s created." % dest_folder)
            os.mkdir(dest_folder)

        for file in segFiles:
            print(file)
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