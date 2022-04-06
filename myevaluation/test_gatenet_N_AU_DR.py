"""
Example command: python -m unittests.sparse_VGG_test
"""
import numpy as np
import torch
import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

import time

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
    
import sys, os
sys.path.insert(0, 'rpg_asynet')

from myconfig.settings import Settings
from mydataloader.dataset import getDataloader
from mymodels.asyn_sparse_vgg import EvalAsynSparseVGGModel, asynSparseVGG
from mymodels.asyn_sparse_vgg_cpp import asynSparseVGGCPP
from mymodels.facebook_sparse_object_det import FBSparseObjectDet
from mymodels.REDnet_sparse_object_det import REDnetSparseObjectDet
from mymodels.custom_REDnetv1_sparse_object_det import customREDnetSparseObjectDet
from mymodels.firenet_sparse_object_det import FirenetSparseObjectDet
from mymodels.yolo_detection import yoloDetect
from mymodels.yolo_detection import nonMaxSuppression
from mytraining.trainer import AbstractTrainer
import rpg_asynet.utils.visualizations as visualizations
import utils.test_util as test_util

#device = torch.device("cuda:0")
device = torch.device("cpu")

class TestObjectDet():

    def __init__(self, args, settings, save_dir='log/N_AU_DR_Results',):
        self.settings = settings
        self.save_dir = save_dir
        self.args = args
        self.multi_processing = args.use_multiprocessing
        self.compute_active_sites = args.compute_active_sites
        self.asyn = args.asyn
        self.yolo_thresh = 0.3

        self.nr_classes = 1
        self.nr_input_channels = 2
        self.sequence_length = 60
        self.output_map = 6 * 8
        self.model_input_size = torch.tensor([self.settings.height, self.settings.width])
        self.total_time = 0

        #settings.model_name = 'fb_sparse_object_det'
        if settings.model_name == 'sparse_REDnet':
            self.model = REDnetSparseObjectDet(self.nr_classes, nr_input_channels=self.nr_input_channels,
                                       small_out_map=(self.settings.dataset_name == 'N_AU_DR')).eval()

        elif settings.model_name == 'custom_sparse_REDnetv1':
            self.model = customREDnetSparseObjectDet(self.nr_classes, nr_input_channels=self.nr_input_channels,
                                       small_out_map=(self.settings.dataset_name == 'N_AU_DR')).eval()

        elif settings.model_name == 'fb_sparse_object_det':
            self.model = FBSparseObjectDet(self.nr_classes, nr_input_channels=self.nr_input_channels,
                                       small_out_map=(self.settings.dataset_name == 'N_AU_DR')).eval()

        elif settings.model_name == 'sparse_firenet':
            self.model = FirenetSparseObjectDet(self.nr_classes, nr_input_channels=self.nr_input_channels,
                                       small_out_map=(self.settings.dataset_name == 'N_AU_DR')).eval()

        self.writer = SummaryWriter(self.save_dir)

    def testSparse(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        # print('Test: %s' % i_test)
        # print('#######################')
        # print('#       New Test      #')
        # print('#######################')

        # ---- Facebook VGG ----

        spatial_dimensions = self.model.spatial_size
        pth = 'log/fb_sparse_obj_det_run3_97/checkpoints/model_step_125.pth'
        self.model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'])

        # ---- Create Input -----
        event_window = 20000

        dataloader = getDataloader(self.settings.dataset_name)
        test_dataset = dataloader(self.settings.dataset_path, 'all', self.settings.height,
                                        self.settings.width, augmentation=False, mode='testing',
                                        nr_events_window=event_window, shuffle=False,
                                        temporal_window=self.settings.temporal_window,
                                        delta_t=self.settings.delta_t)
        self.object_classes = test_dataset.object_classes
        counter = 1
        trackid = 0
        out_dtype = np.dtype([('ts', '<u8'),('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')])
        detected_bounding_boxes = np.empty((0,), dtype = out_dtype)
        
        test = test_dataset.__getitem__(1)


        for i_batch, sample_batched in enumerate(test_dataset):

            events, histogram = sample_batched
            
            if isinstance(events, type(None)):
                break

            if not self.settings.temporal_window:
                print("Getting " + str(event_window) + " events for step: " + str(counter))
            else:
                print("Getting " + str(self.settings.delta_t) + " microseconds worth of events, corresponding to " + str(len(events)) + " events for step: " + str(counter))

            # Histogram for synchronous network
            histogram = torch.from_numpy(histogram[np.newaxis, :, :]).to(device)
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(spatial_dimensions))
            histogram = histogram.permute(0, 2, 3, 1)
            locations, features = AbstractTrainer.denseToSparse(histogram)

            list_spatial_dimensions = [spatial_dimensions.cpu().numpy()[0], spatial_dimensions.cpu().numpy()[1]]
            input_histogram = torch.zeros(list_spatial_dimensions + [2])

            # Detect using synchronous fb network on the whole batch
            tic()
            output = self.model([locations, features])
            self.total_time += toc()
            

            detected_bbox = yoloDetect(output, self.model_input_size.to(output.device),
                   threshold=self.yolo_thresh)

            detected_bbox = nonMaxSuppression(detected_bbox, iou=0.1)
            detected_bbox_long = detected_bbox.long().cpu().numpy()

            # Organizing bounding boxes for saving to npy
            detected_bbox_out = detected_bbox_long.copy()
            detected_bbox_out[:,0] = events[0,2]
            detected_bbox_out[:,7] = trackid
            detected_bbox_out = detected_bbox_out.tolist()
            for i in range(len(detected_bbox_out)):
                detected_bbox_out[i][6] = float("{:.2f}".format((detected_bbox.cpu().detach().numpy())[i,7].copy()))
            trackid += 1

            tuples = tuple(tuple(detected_bbox_out_m) for detected_bbox_out_m in detected_bbox_out)
            for i in range(len(tuples)):
                temp_arr = np.array(tuples[i], dtype=out_dtype)
                detected_bounding_boxes = np.append(detected_bounding_boxes, temp_arr)

            counter += 1

        file_path = os.path.join(self.save_dir, 'result_bounding_boxes' + self.settings.model_name + '.npy')
        np.save(file_path, detected_bounding_boxes)
        avg_time = self.total_time / counter
        print("Average time per " + str(event_window) + " events: " + str(avg_time))


    def testSparseRecurrent(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        # print('Test: %s' % i_test)
        # print('#######################')
        # print('#       New Test      #')
        # print('#######################')

        # ---- Facebook VGG ----

        spatial_dimensions = self.model.spatial_size
        pth = 'log/RNN_TBPTT_trained_run4_93/checkpoints/model_step_15.pth'
        self.model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'])

        # ---- Create Input -----
        event_window = 10000

        dataloader = getDataloader(self.settings.dataset_name)
        test_dataset = dataloader(self.settings.dataset_path, 'all', self.settings.height,
                                        self.settings.width, augmentation=False, mode='testing',
                                        nr_events_window=event_window, shuffle=False)
        self.object_classes = test_dataset.object_classes
        counter = 1
        trackid = 0
        out_dtype = np.dtype([('ts', '<u8'),('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')])
        detected_bounding_boxes = np.empty((0,), dtype = out_dtype)
        
        test = test_dataset.__getitem__(1)
        prev_states = None

        for i_batch, sample_batched in enumerate(test_dataset):

            print("Getting " + str(event_window) + " events for step: " + str(counter))
            events, histogram = sample_batched

            if isinstance(events, type(None)):
                break

            # Histogram for synchronous network
            histogram = torch.from_numpy(histogram[np.newaxis, :, :])
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(spatial_dimensions))
            histogram = histogram.permute(0, 2, 3, 1)
            locations, features = AbstractTrainer.denseToSparse(histogram)

            list_spatial_dimensions = [spatial_dimensions.cpu().numpy()[0], spatial_dimensions.cpu().numpy()[1]]
            input_histogram = torch.zeros(list_spatial_dimensions + [2])

            # Detect using synchronous fb network on the whole batch
            tic()
            if prev_states == None:
                output, new_states = self.model([locations, features, histogram.shape[0]], prev_states)
            else:
                output, new_states = self.model([locations, features, histogram.shape[0]], [prev_states[0].detach(), prev_states[1].detach()])
            prev_states = new_states
            self.total_time += toc()
            

            detected_bbox = yoloDetect(output, self.model_input_size.to(output.device),
                   threshold=self.yolo_thresh)

            detected_bbox = nonMaxSuppression(detected_bbox, iou=0.1)
            detected_bbox_long = detected_bbox.long().cpu().numpy()

            # Organizing bounding boxes for saving to npy
            detected_bbox_out = detected_bbox_long.copy()
            detected_bbox_out[:,0] = events[0,2]
            detected_bbox_out[:,7] = trackid
            detected_bbox_out = detected_bbox_out.tolist()
            for i in range(len(detected_bbox_out)):
                detected_bbox_out[i][6] = float("{:.2f}".format((detected_bbox.cpu().detach().numpy())[i,7].copy()))
            trackid += 1

            tuples = tuple(tuple(detected_bbox_out_m) for detected_bbox_out_m in detected_bbox_out)
            for i in range(len(tuples)):
                temp_arr = np.array(tuples[i], dtype=out_dtype)
                detected_bounding_boxes = np.append(detected_bounding_boxes, temp_arr)

            counter += 1

        file_path = os.path.join(self.save_dir, 'result_bounding_boxes' + self.settings.model_name + '.npy')
        np.save(file_path, detected_bounding_boxes)
        avg_time = self.total_time / counter
        print("Average time per " + str(event_window) + " events: " + str(avg_time))


    def testAsyn(self):
                # ---- Facebook VGG ----
        fb_model = FBSparseObjectDet(self.nr_classes, nr_input_channels=self.nr_input_channels,
                                   small_out_map=(self.settings.dataset_name == 'NCaltech101_ObjectDetection' or
                                                  self.settings.dataset_name == 'N_AU_DR')).eval()
        print(self.settings.dataset_name == 'N_AU_DR')
        spatial_dimensions = fb_model.spatial_size
        pth = 'log/N_AU_DR_trained_run3_best/checkpoints/model_step_125.pth'
        fb_model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'])

        print("initialized sync fb model")

        # ---- Asynchronous VGG ----
        layer_list =  [['C', self.nr_input_channels, 16], ['BNRelu'], ['C', 16, 16],   ['BNRelu'], ['MP'],
                       ['C', 16,   32], ['BNRelu'], ['C', 32, 32],   ['BNRelu'], ['MP'],
                       ['C', 32,   64], ['BNRelu'], ['C', 64, 64],   ['BNRelu'], ['MP'],
                       ['C', 64,  128], ['BNRelu'], ['C', 128, 128], ['BNRelu'], ['MP'],
                       ['C', 128, 256], ['BNRelu'], ['C', 256, 256], ['BNRelu'],
                       ['ClassicC', 256, 256, 3, 2], ['ClassicBNRelu'],
                       ['ClassicFC', 256*self.output_map, 1024], ['ClassicFC', 1024, 576]]

        # self.output_map*(self.nr_classes + 5*self.nr_input_channels)
        asyn_model = EvalAsynSparseVGGModel(nr_classes=self.nr_classes, layer_list=layer_list, 
                                            device=device, input_channels=self.nr_input_channels)
        asyn_model.setWeightsEqual(fb_model)
        print("initialized asyn vgg model")

        # ---- Create Input -----
        event_window = 10000
        events_per_step = 100
        number_of_steps = event_window // events_per_step

        dataloader = getDataloader(self.settings.dataset_name)
        test_dataset = dataloader(self.settings.dataset_path, 'all', self.settings.height,
                                        self.settings.width, augmentation=False, mode='testing',
                                        nr_events_window=event_window, shuffle=False)
        self.object_classes = test_dataset.object_classes
        counter = 1
        trackid = 0
        out_dtype = np.dtype([('ts', '<u8'),('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')])
        detected_bounding_boxes = np.empty((0,), dtype = out_dtype)
        
        test = test_dataset.__getitem__(1)


        for i_batch, sample_batched in enumerate(test_dataset):

            print("Getting " + str(event_window) + " events for step: " + str(counter))
            events, histogram = sample_batched

            # Histogram for synchronous network
            histogram = torch.from_numpy(histogram[np.newaxis, :, :])
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(spatial_dimensions))
            histogram = histogram.permute(0, 2, 3, 1)
            locations, features = AbstractTrainer.denseToSparse(histogram)

            list_spatial_dimensions = [spatial_dimensions.cpu().numpy()[0], spatial_dimensions.cpu().numpy()[1]]
            input_histogram = torch.zeros(list_spatial_dimensions + [2])

            # Detect using synchronous fb network on the whole batch
            tic()
            fb_output = fb_model([locations, features])
            self.total_time += toc()
            

            fb_detected_bbox = yoloDetect(fb_output, self.model_input_size.to(fb_output.device),
                   threshold=self.yolo_thresh)

            fb_detected_bbox = nonMaxSuppression(fb_detected_bbox, iou=0.3)
            fb_detected_bbox_long = fb_detected_bbox.long().cpu().numpy()

            # Organizing bounding boxes for saving to npy
            fb_detected_bbox_out = fb_detected_bbox_long.copy()
            fb_detected_bbox_out[:,0] = events[0,2]
            fb_detected_bbox_out[:,7] = trackid
            fb_detected_bbox_out = fb_detected_bbox_out.tolist()
            for i in range(len(fb_detected_bbox_out)):
                fb_detected_bbox_out[i][6] = float("{:.2f}".format((fb_detected_bbox.cpu().detach().numpy())[i,7].copy()))
            trackid += 1

            tuples = tuple(tuple(fb_detected_bbox_out_m) for fb_detected_bbox_out_m in fb_detected_bbox_out)
            for i in range(len(tuples)):
                temp_arr = np.array(tuples[i], dtype=out_dtype)
                detected_bounding_boxes = np.append(detected_bounding_boxes, temp_arr)

            if i_batch == 0:
                with torch.no_grad():
                    # Fill asynchronous input representation and compute
                    asyn_locations, input_histogram = asyn_model.generateAsynInput(events, spatial_dimensions,
                                                                           original_shape=[self.settings.height, self.settings.width])
                    x_asyn = [None] * 5
                    x_asyn[0] = asyn_locations[:, :2].to(device)
                    x_asyn[1] = input_histogram.to(device)
                    # Detect using async network
                    tic()
                    asyn_output1 = asyn_model.forward(x_asyn)
                    toc()
                    asyn_output = asyn_output1[1].view([-1] + [6,8] + [(self.nr_classes + 5*self.nr_input_channels)])
                    asyn_detected_bbox = yoloDetect(asyn_output.float(), self.model_input_size.to(asyn_output.device),
                           threshold=0.3)
                    asyn_detected_bbox = nonMaxSuppression(asyn_detected_bbox, iou=0.6)
                    asyn_detected_bbox = asyn_detected_bbox.long().cpu().numpy()
                    
                old_events = np.copy(events)
                continue
            

            # Try to detect using asynchronous model  
            with torch.no_grad():
                for i_sequence in range(number_of_steps):
                    #Generate input reprensetation for asynchrnonous network
                    
                    new_batch_events = events[(events_per_step*i_sequence):(events_per_step*(i_sequence + 1)), :]
                    update_locations, new_histogram = asyn_model.generateAsynInput(new_batch_events, spatial_dimensions,
                                                                               original_shape=[self.settings.height, self.settings.width])

                    input_histogram = input_histogram + new_histogram
                    x_asyn = [None] * 5
                    x_asyn[0] = update_locations[:, :2].to(device)
                    x_asyn[1] = new_histogram.to(device)
                    # Detect using async network
                    tic()
                    asyn_output1 = asyn_model.forward(x_asyn)
                    toc()
                    asyn_output = asyn_output1[1].view([-1] + [6,8] + [(self.nr_classes + 5*self.nr_input_channels)])
                    asyn_detected_bbox = yoloDetect(asyn_output.float(), self.model_input_size.to(asyn_output.device),
                           threshold=0.3)
                    asyn_detected_bbox = nonMaxSuppression(asyn_detected_bbox, iou=0.3)
                    asyn_detected_bbox = asyn_detected_bbox.long().cpu().numpy()

            counter += 1

            if counter % 5 == 0:
                print("saving")

                #file_path = os.path.join(self.save_dir, 'test_results.pth')
                #torch.save({'state_dict': fb_model.state_dict()}, file_path)
        file_path = os.path.join(self.save_dir, 'result_bounding_boxes.npy')
        np.save(file_path, detected_bounding_boxes)
        avg_time = self.total_time / counter
        print("Average time per " + str(event_window) + " events: " + str(avg_time))


def main():
    parser = argparse.ArgumentParser(description='Test network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--save_dir', help='Path to save location')
    parser.add_argument('--representation', default="")
    parser.add_argument('--use_multiprocessing', help='If multiprocessing should be used', action='store_true')
    parser.add_argument('--compute_active_sites', help='If active sites should be calculated', action='store_true')
    parser.add_argument('--asyn', help='If asynchronous network should be used', action='store_true')

    args = parser.parse_args()
    settings_filepath = args.settings_file
    save_dir = args.save_dir
    
    settings = Settings(settings_filepath, generate_log=False)

    tester = TestObjectDet(args, settings, save_dir)
    tester.testSparseRecurrent()


if __name__ == "__main__":
    main()
