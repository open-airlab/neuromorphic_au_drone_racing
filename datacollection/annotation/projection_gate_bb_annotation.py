import cv2
import os
import json
import shutil
import numpy as np
from shutil import copyfile
import json
import math
from scipy.spatial.transform import Rotation as R

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir) 
from mydataloader.prophesee import npy_events_tools

from quaternion_utils import Quaternion

def quat2eul(Q):
    
        w = Q.w
        x = Q.x
        y = Q.y
        z = Q.z

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        X = math.atan2(t0, t1)
        Y = math.asin(t2)
        Z = math.atan2(t3, t4)

        return X, Y, Z
    
def eul2quat(rpy=None):
    yaw = rpy[0]
    pitch = rpy[1]
    roll = rpy[2]

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    out = Quaternion([qw, qx, qy, qz])
    return out


class GateBoundingBoxAnnotation:
    """ Annotate gates for training using back projection

        This class is a converter to 3D coordinates of gates to 2D pixel coordinates.
    """

    def __init__(self):

        self.resolution = [240, 180]
        cameraFOV = 90
        focal_length = self.resolution[0] / (2*math.tan(cameraFOV * math.pi / 360))
        centerX = self.resolution[0] / 2
        centerY = self.resolution[1] / 2
        self.intrinsics = np.array([focal_length, focal_length, centerX, centerY])
        self.trackid_counter = 0

        # Static gate coordinates
        self.gate_positions = np.array([[[3302, -280, 306.5], [3302, -120, 306.5], [3302, -120, 146.5], [3302, -280, 146.5]],
                                        [[2880, 302, 306.5], [2720, 302, 306.5], [2720, 302, 146.5], [2880, 302, 146.5]],
                                        [[2298, -120, 306.5], [2298, -280, 306.5], [2298, -280, 146.5], [2298, -120, 146.5]],
                                        [[2720, -702, 306.5], [2880, -702, 306.5], [2880, -702, 146.5], [2720, -702, 146.5]]])

        # Camera matrix
        self.camera_matrix = np.array([[self.intrinsics[0], 0, self.intrinsics[2], 0,], [0, self.intrinsics[1], self.intrinsics[3], 0], [0, 0, 1, 0]])
        self.film_matrix = np.array([[self.intrinsics[0], 0, 0, 0], [0, self.intrinsics[1], 0, 0], [0, 0, 1, 0]])
        # Estimated offsed between camera and UAV's COM
        self.camera_position_offset = np.array([0.0, 0.0, 0.0]) # in UAV body frame
        self.camera_rotation_offset = np.array([0.0, 0.0, 0.0])
        self.camera_offset_quaternion = (eul2quat([self.camera_rotation_offset[2], self.camera_rotation_offset[1], self.camera_rotation_offset[0]])).inverse
        #self.camera_offset_quaternion = quatinv(eul2quat([camera_rotation_offset(3) camera_rotation_offset(2) camera_rotation_offset(1)]));
        
        # Quaternion function for ease of use:


    def projection(self, drone_x, drone_y, drone_z, drone_qw, drone_qx, drone_qy, drone_qz):
        
        results = []
        num = 0

        for i,  gate in enumerate(self.gate_positions):
            
            unreal_offset = np.array([27.608, -2.09, 0.78])

            
            # Drone position and orientation
            drone_position = np.array([[drone_x, drone_y, drone_z]])
            drone_position_3D = unreal_offset - drone_position
            drone_quaternion = Quaternion([drone_qw, drone_qx, drone_qy, drone_qz])
            drone_orientation = np.array([quat2eul(drone_quaternion.inverse)])
    
            # Combines rotation of the drone and rotation of the camera wrt the drone (camera_orientation = drone_orientation + camera_rotation_offset) 
            camera_quaternion = drone_quaternion * self.camera_offset_quaternion
            camera_orientation = np.array([quat2eul(camera_quaternion.inverse)])
            
            # Transforming gate corners' position into drone's body position
            corners_translation = np.empty((4,3))
            image_corner = np.empty((4,2))
            calc_bb = True
            for j, corner in enumerate(gate):
                #corners_translation[j, :] = camera_quaternion.rotate((corner/100 - drone_position_3D)[0])
                
                pose_matrix = np.zeros((4,4))
                
                pose_matrix[0:3, 0:3] = camera_quaternion.rotation_matrix
                homogen_rot_matrix = np.copy(pose_matrix)
                homogen_rot_matrix[3,3] = 1
                pose_matrix[0:3, 3] = drone_position_3D
                pose_matrix[3,3] = 1
                extrinsic_matrix = pose_matrix
                
                #rot_matrix = (camera_quaternion.rotation_matrix).T
                #translation_vector = -(np.matmul(drone_position_3D, rot_matrix))
                #extrinsic_matrix = np.zeros((4,4))
                #extrinsic_matrix[0:3, 0:3] = rot_matrix
                #extrinsic_matrix[0:3, 3] = translation_vector
                #extrinsic_matrix[3,3] = 1
    
                #projected_point = np.matmul(np.matmul(self.camera_matrix, np.linalg.inv(extrinsic_matrix)), np.append(corner/100, 1).T)
                point_in_camera_coords = np.matmul(homogen_rot_matrix, np.append(corner/100 - drone_position_3D, 1).T)
                fixed_point_in_camera_coords = np.array([-point_in_camera_coords[1], -point_in_camera_coords[2], point_in_camera_coords[0], 1])
                if fixed_point_in_camera_coords[2] < 0:
                    calc_bb = False
                    continue
                point_in_image_coords = np.matmul(self.camera_matrix, fixed_point_in_camera_coords)
                
                image_corner[j, :] = np.array([point_in_image_coords[0]/point_in_image_coords[2], point_in_image_coords[1]/point_in_image_coords[2]])
                #projected_point = self.camera_matrix(np.matmul(corner/100, rot_matrix)
                #image_corner = np.array([projected_point[0]/projected_point[2], projected_point[1]/projected_point[2]])
                
                if not (0 < image_corner[j, 0] < self.resolution[0]) and not (0 < image_corner[j,1] < self.resolution[1]):
                    calc_bb = False
                    break
                
            if calc_bb == True:
                print("Got one")
                max_x = np.max(image_corner[:,0])
                min_x = np.min(image_corner[:,0])
                max_y = np.max(image_corner[:,1])
                min_y = np.min(image_corner[:,1])
                x = min_x
                y = min_y
                w = max_x - min_x
                h = max_y - min_y
                class_id = 1
                confidence = 1
                track_id = self.trackid_counter
                results.append((x, y, w, h, class_id, confidence, track_id))
                self.trackid_counter += 1
                num += 1

        return results, num

    def drawPolygon(self, img, points, color):
        line_thickness = 2
        points = points.round()
        for i in range(points.shape[0]-1):
            x1, y1 = int(points[i][0]), int(points[i][1])
            x2, y2 = int(points[i+1][0]), int(points[i+1][1])
            cv2.line(img, (x1, y1), (x2, y2), color, thickness=3)



    def run(self, source_folder, dest_folder, lookup=None):

        print("[*] Source folder: %s" % source_folder)
        output_dtype = np.dtype([('ts', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')])
        
        # Get all attribute files
        attrFiles = []
        for file in os.listdir(source_folder):
            if file.endswith(".npy"):
                attrFiles.append(file)
                  

        num_of_samples = len(attrFiles)
        print("[*] Num of samples found: %d" % num_of_samples)


        # Create output folder.
        if not os.path.isdir(dest_folder):
            print("%s created." % dest_folder)
            os.mkdir(dest_folder)
            os.mkdir(os.path.join(dest_folder, 'images'))

        output = np.empty((0,), dtype = output_dtype)
        for file in attrFiles:
            
            attr = np.load(os.path.join(source_folder, file))
            for i, ts in enumerate(attr[0,:]):
                            
                results, num = self.projection(attr[1,i], attr[2,i], attr[3,i], attr[4,i],
                                          attr[5,i], attr[6,i], attr[7,i])
                if num > 0:
                    if num == 1:
                        list_res = list(results[0])
                        list_res.insert(0,ts)
                        res = tuple(list_res)
                        
                        temp_arr = np.array(res, dtype=output_dtype)
                        output = np.append(output, temp_arr)
                        
                    elif num > 1:
                        for i in range(num):
                        
                            list_res = list(results[i])
                            list_res.insert(0,ts)
                            res = tuple(list_res)
                            
                            temp_arr = np.array(res, dtype=output_dtype)
                            output = np.append(output, temp_arr)
                        
                        
            np.save(os.path.join(dest_folder, 'test.npy'), output)
                    


def main():
    source_folder = os.path.join(parentdir, 'source_data')
    dest_folder = os.path.join(parentdir, 'dest_data')
    
    bb = GateBoundingBoxAnnotation()
    bb.run(source_folder, dest_folder)
    
if __name__ == "__main__":
    main()