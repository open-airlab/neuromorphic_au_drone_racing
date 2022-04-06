import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import numpy as np
import airsim
from datetime import datetime
import time
import cv2
import matplotlib.pyplot as plt
import argparse
import sys, signal
import pandas as pd
import pickle
from event_simulator import *
from mydataloader.prophesee import dat_events_tools

parser = argparse.ArgumentParser(description="Simulate event data from AirSim")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--height", type=int, default=180)
parser.add_argument("--width", type=int, default=240)
parser.add_argument("--segment", action='store_true')
parser.add_argument("--attr", action='store_true')

#dtype = np.dtype([('ts', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')])


class AirSimEventGen:
    def __init__(self, W, H, debug=False, segment=False, attr=False):

        self.ev_sim = EventSimulator(H, W)

        self.image_request = airsim.ImageRequest(
            "cam", airsim.ImageType.Scene, False, False
        )

        self.W = W
        self.H = H
        self.segment = segment
        self.attr = attr

        self.image_list = []

        self.client = airsim.VehicleClient()
        self.client.confirmConnection()
        self.init = True
        self.start_ts = None

        if self.segment:
            found = self.client.simSetSegmentationObjectID("[\w]*", 255, True)
            found = self.client.simSetSegmentationObjectID("Segment_gate", 0, True);
            print(found)
            found = self.client.simSetSegmentationObjectID("segment_gate2_7", 0, True);
            print(found)
            found = self.client.simSetSegmentationObjectID("segment_gate3_5", 0, True);
            print(found)
            found = self.client.simSetSegmentationObjectID("segment_gate4", 0, True);
            print(found)
            self.segment_request = airsim.ImageRequest("cam", airsim.ImageType.Segmentation, False, False)
            self.segment_list = []


        self.rgb_image_shape = [H, W, 3]
        self.debug = debug

        # Setup .dat event file
        self.tstart = str(int(time.time()))
        self.event_file = self.setup_event_file()

        if debug:
            self.fig, self.ax = plt.subplots(1, 1)

        ## Attribute collection
        self.attrFrequency = 5 # Hz
        if self.attr:
            self.counter = 0
            self.singleDroneAttribute = np.zeros(14)
            self.droneAttributes = np.zeros([14])

    def setup_event_file(self):
        self.date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.event_filename = self.date + '_' + self.tstart + '_' + str(self.ev_sim.tol) + '_td.dat'
        event_file = dat_events_tools.write_header(self.event_filename, self.H, self.W)
        return event_file

    def visualize_events(self, event_img):
        event_img = self.convert_event_img_rgb(event_img)
        self.ax.cla()
        self.ax.imshow(event_img, cmap="viridis")
        plt.draw()
        plt.pause(0.001)

    def convert_event_img_rgb(self, image):
        image = image.reshape(self.H, self.W)
        out = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        out[:, :, 0] = np.clip(image, 0, 1) * 255
        out[:, :, 2] = np.clip(image, -1, 0) * -255

        return out

    def save_to_files(self):
        if self.segment:
            np.save(self.date + '_' + self.tstart + '_' + str(self.ev_sim.tol) + '_segmentation.npy', self.segment_list)
            del self.segment_list
            self.segment_list = []
        elif self.attr:
            np.save(self.date + '_' + self.tstart + '_attr.npy', self.droneAttributes[:,1:])
            self.droneAttributes = np.zeros([14])
        del self.image_list
        self.image_list = []
        del self.ev_sim
        self.ev_sim = EventSimulator(self.H, self.W)


    def remove_files_and_restart(self):
        if self.segment:
            del self.segment_list
            self.segment_list = []
        elif self.attr:
            self.droneAttributes = np.zeros([14])
        del self.image_list
        self.image_list = []
        del self.ev_sim
        self.ev_sim = EventSimulator(self.H, self.W)



    def _stop_event_gen(self, signal, frame):
        print("\nCtrl+C received. Stopping event sim...")
        sys.exit(0)

    ## Attribute collection
    # Attributes are collected in a .npy file with the following formatting:
    # [ts, pos_x, pos_y, pos_z, att_w, att_x, att_y, att_z, 
    # linvel_x, linvel_y, linvel_z, angvel_x, angvel_y, angvel_z]
    # We might assume that the gates are static and does not change position and pose
    def collectData(self, ts):
        self.singleDroneAttribute[0] = ts
        kinematics = self.client.simGetGroundTruthKinematics('PX4')
        pos = kinematics.position
        ori = kinematics.orientation
        linvel = kinematics.linear_velocity
        angvel = kinematics.angular_velocity


        self.singleDroneAttribute[1:8] = [pos.x_val, pos.y_val, pos.z_val,
                                          ori.w_val, ori.x_val, ori.y_val, ori.z_val]
        self.singleDroneAttribute[8:14] = [linvel.x_val, linvel.y_val, linvel.z_val,
                                           angvel.x_val, angvel.y_val, angvel.z_val]

        self.droneAttributes = np.c_[self.droneAttributes, self.singleDroneAttribute]
        

    def saveAttrToNpy(self):
        np.save(self.attrFilename, self.droneAttributes)


if __name__ == "__main__":
    args = parser.parse_args()

    event_generator = AirSimEventGen(args.width, args.height, debug=args.debug, segment=args.segment, attr=args.attr)
    number_of_trials = 50

    signal.signal(signal.SIGINT, event_generator._stop_event_gen)

    for i in range(number_of_trials):
        reshape_error = False
        t_start = time.time()
        t_start_segment = t_start
        print("here we go")
        # First collect images and segmentation images if enabled
        while (time.time() - t_start) < 60:

            #t2 = time.time_ns()
            response = event_generator.client.simGetImages([event_generator.image_request])

            if event_generator.init:
                event_generator.start_ts = response[0].time_stamp
                event_generator.init = False
            try:
                img = np.reshape(
                    np.fromstring(response[0].image_data_uint8, dtype=np.uint8),
                    event_generator.rgb_image_shape,
                )


                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Add small number to avoid issues with log(I)
                img = cv2.add(img, 0.001)

                if event_generator.debug:
                    cv2.imshow("debug", img)
                    cv2.waitKey(1)



                event_generator.image_list.append((response[0].time_stamp, img))

                if event_generator.segment and (time.time() - t_start_segment) > (1/event_generator.attrFrequency):
                    t_start_segment = time.time()
                    response_segment = event_generator.client.simGetImages([event_generator.segment_request])
                    img_seg = np.reshape(
                        np.fromstring(response_segment[0].image_data_uint8, dtype=np.uint8),
                        event_generator.rgb_image_shape,
                    )

                    img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY).astype(np.float32)

                    if event_generator.debug:
                        print(img_seg)
                        event_generator.ax.imshow(img_seg, cmap="viridis")
                        plt.draw()
                        plt.pause(0.001)

                    sync_timestamp = (response_segment[0].time_stamp - event_generator.start_ts) * 1e-3
                    event_generator.segment_list.append((sync_timestamp, img_seg))
                
                elif event_generator.attr and (time.time() - t_start_segment) > (1/event_generator.attrFrequency):
                    t_start_segment = time.time()
                    event_generator.collectData((time.time() - t_start) * 1e-3)

                #deltat = time.time_ns() - t2
                #print("total: ", deltat/1000000)
            except Exception as e:
                print("reshape error. Dropping sequence")
                print(e)
                reshape_error = True
                break



        if reshape_error == False:
            # Next run event simulator on collected images
            event_generator.init = True
            for img in event_generator.image_list:

                if event_generator.init:
                    event_generator.start_ts = img[0]
                    event_generator.init = False
                    continue

                ts_delta = (img[0] - event_generator.start_ts) * 1e-3

                # Event sim keeps track of previous image automatically
                event_img, events = event_generator.ev_sim.image_callback(img[1], ts_delta)

                #tnew = time.time_ns()
                if events is not None and events.shape[0] > 0:
                    events['timestamp'] = (events['timestamp']*1000000).astype(int)
                    dat_events_tools.write_event_buffer(event_generator.event_file, events)
                    if event_generator.debug:
                        event_generator.visualize_events(event_img)

            event_generator.save_to_files()
            event_generator.event_file.close()

            event_generator.init = True
            if i != number_of_trials-1:
                event_generator.event_file = event_generator.setup_event_file()
        
        else:
            event_generator.remove_files_and_restart()
            os.remove(event_generator.event_file.name)
            event_generator.init = True
            event_generator.event_file = event_generator.setup_event_file()


