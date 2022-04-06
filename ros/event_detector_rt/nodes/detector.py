#!/usr/bin/env python3

# Standard stuff
import numpy as np
import cv2 as cv
import torch
import time
from collections import deque
import rospy
from dvs_msgs.msg import Event, EventArray
from event_detector_rt.msg import BoundingBox, BoundingBoxes
from event_preprocessor.msg import HistogramStamped

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Deep learning stuff
from models.facebook_sparse_object_det import FBSparseObjectDet
from models.sparse_RNN_object_det import SparseRNNObjectDet
from models.dense_object_det import DenseObjectDet
from models.yolo_detection import yoloDetect
from models.yolo_detection import nonMaxSuppression

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
        return tempTimeInterval

def tic():
    # Records a time in TicToc, marks thex_asyn[1].unsqueeze(0) beginning of a time interval
    toc(False)

print(torch.__version__)
print("CUDA available: " + str(torch.cuda.is_available()))

# device = torch.device("cuda:0") #if graphic card + cuda is available 
device = torch.device("cpu")

class RealTimeDetector():

    def __init__(self, debug=False):

        ### ROS initialization and parameters ###
        
        # Initialize ROS node
        rospy.init_node('GateDetector', anonymous=True, disable_signals=True)

        # Publisher
        self.debug = debug
        self.debug_counter = 0

        self.total_elapse_time = 0;
        self.forward_pass_counter = 0;

        self.bounding_box_pub = rospy.Publisher('/detections/bb', BoundingBoxes, queue_size=1)
        if self.debug:
            self.bridge = CvBridge()
            self.debug_img_pub = rospy.Publisher('/detections/debug', Image, queue_size=1)

        # Subscriber
        self.hist_received = False;
        self.hist_sub = rospy.Subscriber('/event_preprocessor/event_histogram', HistogramStamped, self.histCallback, queue_size=1, buff_size=65536000, tcp_nodelay=True)

        ### Object detector initialization and parameters ###
        self.settings = rospy.get_param('GateDetector')

        self.yolo_thresh = self.settings['model']['yolo_threshold']
        self.yolo_iou = self.settings['model']['yolo_iou']

        self.event_window = self.settings['dataset']['nr_events_window']
        self.nr_classes = 1
        self.nr_input_channels = 2
        self.output_map = 5 * 7
        self.height = self.settings['dataset']['height']
        self.width = self.settings['dataset']['width']
        self.model_input_size = torch.tensor([191, 255])
        self.model_layers = self.settings['model']['model_name'].split('_')[0]
        self.model_type = self.settings['model']['model_name'].split('_')[1]

        self.prev_states = None
        self.event_queue = deque(maxlen=self.event_window)
        self.inference_running = False

        if self.settings['model']['model_name'] == 'sparse_VGG':
            self.model = FBSparseObjectDet(self.nr_classes, nr_input_channels=self.nr_input_channels,
                                       small_out_map=True).eval()
            pth = self.settings['checkpoint']['pretrained_sparse_vgg']
            self.model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'])

        elif self.settings['model']['model_name'] == 'sparse_RNN':
            self.model = SparseRNNObjectDet(self.nr_classes, nr_input_channels=self.nr_input_channels,
                                       small_out_map=True).eval()
            pth = self.settings['checkpoint']['pretrained_sparse_rnn']
            self.model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'], strict=True)

        elif self.settings['model']['model_name'] == 'dense_VGG':
            self.model = DenseObjectDet(self.nr_classes, in_c=self.nr_input_channels,
                                       small_out_map=True).eval()
            pth = self.settings['checkpoint']['pretrained_dense_vgg']
            self.model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'], strict=True)
        
        img_pos = np.zeros((self.height * self.width,), dtype="float32")
        img_neg = np.zeros((self.height * self.width,), dtype="float32")
        self.histogram =  np.stack([img_neg, img_pos], -1).reshape((180, 240, 2))

        print("[Event Detector]: Initialized!")


    @staticmethod
    def denseToSparse(dense_tensor):
        """
        Converts a dense tensor to a sparse vector.

        :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
        :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
        :return features: NumberOfActive x FeatureDimension
        """
        non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
        locations = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)

        select_indices = non_zero_indices.split(1, dim=1)
        features = torch.squeeze(dense_tensor[select_indices], dim=-2)

        return locations, features


    def forward(self, histogram):
        
        if self.model_layers == 'sparse':
            # Sparse histogram
            histogram = torch.from_numpy(histogram[np.newaxis, :, :]).to(device)
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), 
                                                        torch.Size(self.model.spatial_size))
            histogram = histogram.permute(0, 2, 3, 1)
            locations, features = self.denseToSparse(histogram)

        elif self.model_layers == 'dense':
            # Dense histogram
            histogram = torch.from_numpy(histogram[np.newaxis, :, :]).to(device)
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size))
        else: 
            raise TypeError("Model layers not recognized. Check model_name in config file.")

        print("device usage = ", histogram.device)

        if self.model_type == 'VGG':
            if self.model_layers == 'sparse':
                with torch.no_grad():
                    output = self.model([locations, features])
            elif self.model_layers == 'dense':
                 with torch.no_grad():
                    output = self.model(histogram)

        elif self.model_type == 'RNN':
            with torch.no_grad():
                output, self.prev_states = self.model([locations, features, histogram.shape[0]], self.prev_states)
        else:
            print(self.model_type)
            raise TypeError("Model type not recognized, Check model_name in config file")

        
  
        detected_bbox = yoloDetect(output, self.model_input_size.to(output.device),
               threshold=self.yolo_thresh)

        detected_bbox = nonMaxSuppression(detected_bbox, iou=self.yolo_iou)
        detected_bbox_long = detected_bbox.long().cpu().numpy()

        return detected_bbox_long
        #queue.put(detected_bbox_long)

        # 26.860.000 events
        # 1763 packages
        # 29 packages per second. Each second 29 ms is spent on the deque.
        # 15.235 events per package

        # 447.000 events/second
        # 45 windows/second = 22ms per window

    def histCallback(self, data):
        # print(len(data.positive))
        img_pos =  np.float32(data.positive)
        img_neg =  np.float32(data.negative) 
        self.histogram =  np.stack([img_neg, img_pos], -1).reshape((180, 240, 2))
        self.hist_received = True


    def evCallback(self, data):

        self.event_queue.extend(data.events)


    def generateHistogram(self):

        event_copy = self.event_queue.copy()
        x = np.array([ev.x for ev in event_copy])
        y = np.array([ev.y for ev in event_copy])
        p = np.array([ev.polarity for ev in event_copy])

        img_pos = np.zeros((self.height * self.width,), dtype="float32")
        img_neg = np.zeros((self.height * self.width,), dtype="float32")

        np.add.at(img_pos, x[p == True] + self.width * y[p == True], 1)
        np.add.at(img_neg, x[p == False] + self.width * y[p == False], 1)

        histogram = np.stack([img_neg, img_pos], -1).reshape((self.height, self.width, 2))
        return histogram


    def inferenceLoop(self):

        torch.set_num_threads(4)
        # while not rospy.is_shutdown():
        while self.hist_received == True:

            # if len(self.event_queue) == self.event_window:
                # Create histogram (time 4ms)
                # histogram = self.generateHistogram()

            # Run network (time 50ms)
            print("Forward: ")
            tic()
            bboxes = self.forward(self.histogram)
            elapse_time = toc()
            self.total_elapse_time += elapse_time
            self.forward_pass_counter += 1
            print("running average prediction time = ", self.total_elapse_time/self.forward_pass_counter)
            GPUtil.showUtilization()

            
            if self.debug: # Debug time: 170ms

                # Construct histogram into image
                pos_hist = self.histogram[:,:,0]
                neg_hist = self.histogram[:,:,1]
                
                combined_hist = pos_hist + neg_hist # for calculating moment

                img = 255 * np.ones(shape=(self.height,self.width, 3), dtype=np.uint8)
                img_combined_hist = 0 * np.ones(shape=(self.height,self.width, 1), dtype=np.uint8)

                for x in range(self.width):
                    for y in range(self.height):
                        if pos_hist[y, x] > 0:
                            img[y, x, :] = 0
                            img[y, x, 0] = 255
                            img[y, x, 2] = 0

                        if neg_hist[y,x] > 0:
                            img[y, x, :] = 0
                            img[y, x, 0] = 0
                            img[y, x, 2] = 255

                        if combined_hist[y,x] > 0:
                            img_combined_hist[y, x, 0] = 255

                #img = [255 for pix in pos_hist if pix > 0]
                #img = [0 for pix in neg_hist if pix > 0]

                # Overlay bounding box onto image                   
                for i in range(bboxes.shape[0]):

                    pt1 = (int(bboxes[i][1]), int(bboxes[i][2]))
                    size = (int(bboxes[i][3]), int(bboxes[i][4]))
                    pt2 = (pt1[0] + size[0], pt1[1] + size[1])
                    score = 1 # TODO
                    class_id = "Gate"
                    class_name = "Gate"
                    color = (0,255,0)
                    center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv.rectangle(img, pt1, pt2, color, 1)
                    # cv.circle(img, (pt1[0] + size[0]//2, pt1[1] + size[1]//2), 5, (0,225,0), -1)
                    cv.circle(img, (center[0], center[1]), 5, (0,225,0), -1)
                    # cv.circle(img, (pt1[0], pt1[1]), 5, (0,225,0), -1) #bbox give the top left corner of the bounding box

                    cv.putText(img, class_name, (center[0], pt2[1] - 1), cv.FONT_HERSHEY_SIMPLEX, 0.5, color)
                    if score != 1:
                        cv.putText(img, str(score), (center[0], pt1[1] - 1), cv.FONT_HERSHEY_SIMPLEX, 0.5, color)

                # Moment implement:
                # BEGIN calculate image distance

                M = cv.moments(img_combined_hist)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    # print("cx = ", cx, "; cy = ", cy)
                    # cv.rectangle(img,(cx - 50, cy - 50), (cx + 50, cy + 50), (0,255,255), 1)
                    # cv.circle(img, (cx, cy), 5, (0,225,255), -1)
                # cv.rectangle(img,(120 - 50, 90 - 50), (120 + 50, 90 + 50), (0,255,255), 1)
                # cv.circle(img, (120, 90), 5, (0,225,255), -1)

                # Publish image
                debug_img = Image()
                debug_img.height = self.height
                debug_img.width = self.width
                debug_img.data = img.tostring()
                debug_img.encoding = "rgb8"
                debug_img.is_bigendian = 0
                debug_img.step = len(debug_img.data) // debug_img.height

                self.debug_img_pub.publish(debug_img)

            # Publish bounding box (time negligible)
            if bboxes.any():

                ros_bboxes = BoundingBoxes()
                for bbox in bboxes:

                    conf = 1
                    center_x = int(bbox[1] + bbox[3]//2)
                    center_y = int(bbox[2] + bbox[4]//2)
                    # ros_bboxes.bounding_boxes.append(BoundingBox('Gate', conf, bbox[1], bbox[2], bbox[3], bbox[4]))
                    ros_bboxes.bounding_boxes.append(BoundingBox('Gate', conf, center_x, center_y, bbox[3], bbox[4]))
                    self.bounding_box_pub.publish(ros_bboxes)

def main():

    detector = RealTimeDetector(debug=True)
    detector.inferenceLoop()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


if __name__ == "__main__":
    main()
