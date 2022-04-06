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
from mymodels.yolo_detection import yoloDetect
from mymodels.yolo_detection import nonMaxSuppression
from mytraining.trainer import AbstractTrainer
import rpg_asynet.utils.visualizations as visualizations
import utils.test_util as test_util

# DEVICE = torch.device("cuda:0")
device = torch.device("cpu")

class TestSparseVGG():

    def __init__(self, args, settings, save_dir='log/PropheseeResults',):
        self.settings = settings
        self.save_dir = save_dir
        self.args = args
        self.multi_processing = args.use_multiprocessing
        self.compute_active_sites = args.compute_active_sites

        self.nr_classes = 2
        self.nr_input_channels = 2
        self.sequence_length = 60
        self.output_map = 6 * 8
        self.model_input_size = torch.tensor([self.settings.height, self.settings.width])
        self.asyn = 0


        self.writer = SummaryWriter(self.save_dir)

    def test_sparse_VGG(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        # print('Test: %s' % i_test)
        # print('#######################')
        # print('#       New Test      #')
        # print('#######################')

        # ---- Facebook VGG ----
        fb_model = FBSparseObjectDet(self.nr_classes, nr_input_channels=self.nr_input_channels,
                                   small_out_map=(self.settings.dataset_name == 'NCaltech101_ObjectDetection')).eval()
        spatial_dimensions = fb_model.spatial_size
        pth = 'log/prophesee_trained_200epochs/checkpoints/model_step_181.pth'
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
        event_window = 25000
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


        for i_batch, sample_batched in enumerate(test_dataset):

            print("Getting 25000 events for step: " + str(counter))
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
            toc()

            fb_detected_bbox = yoloDetect(fb_output, self.model_input_size.to(fb_output.device),
                   threshold=0.3)

            fb_detected_bbox = nonMaxSuppression(fb_detected_bbox, iou=0.6)
            fb_detected_bbox = fb_detected_bbox.long().cpu().numpy()

            # Organizing bounding boxes for saving to npy
            fb_detected_bbox_out = fb_detected_bbox.copy()
            fb_detected_bbox_out[:,0] = events[0,2]
            fb_detected_bbox_out[:,7] = trackid
            trackid += 1

            tuples = tuple(tuple(fb_detected_bbox_out_m.tolist()) for fb_detected_bbox_out_m in fb_detected_bbox_out)
            for i in range(len(tuples)):
                temp_arr = np.array(tuples[i], dtype=out_dtype)
                detected_bounding_boxes = np.append(detected_bounding_boxes, temp_arr)

            if self.asyn:

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
                        #old_batch_evs = old_events[events_per_step*(i_sequence+1):,:]
                        #new_batch_evs = events[0:events_per_step*(i_sequence+1),:]
                        #current_batch_evs = np.r_[old_batch_evs, new_batch_evs]
                        
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
                        asyn_detected_bbox = nonMaxSuppression(asyn_detected_bbox, iou=0.6)
                        asyn_detected_bbox = asyn_detected_bbox.long().cpu().numpy()
            
            """
            batch_one_mask = locations[:, -1] == 0
            vis_locations = locations[batch_one_mask, :2]
            features = features[batch_one_mask, :]
            vis_detected_bbox = fb_detected_bbox[fb_detected_bbox[:, 0] == 0, 1:-2].astype(np.int)

            image = visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), self.model_input_size,
                                                      features=features.cpu().numpy())

            image = visualizations.drawBoundingBoxes(image, vis_detected_bbox[:, :-1],
                                                    class_name=[self.object_classes[i]
                                                                for i in fb_detected_bbox[:, -1]],
                                                     ground_truth=False, rescale_image=True)

            self.writer.add_image('FB', image, counter, dataformats='HWC')

            
            batch_one_mask = locations[:, -1] == 0
            vis_locations = locations[batch_one_mask, :2]
            features = features[batch_one_mask, :]
            vis_detected_bbox = asyn_detected_bbox[asyn_detected_bbox[:, 0] == 0, 1:-2].astype(np.int)

            image = visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), self.model_input_size,
                                                      features=features.cpu().numpy())

            image = visualizations.drawBoundingBoxes(image, vis_detected_bbox[:, :-1],
                                                    class_name=[self.object_classes[i]
                                                                for i in asyn_detected_bbox[:, -1]],
                                                     ground_truth=False, rescale_image=True)

            self.writer.add_image('ASYN', image, counter, dataformats='HWC')
            """

            counter += 1

            if counter % 5 == 0:
                print("saving")

                file_path = os.path.join(self.save_dir, 'test_results.pth')
                torch.save({'state_dict': fb_model.state_dict()}, file_path)
        file_path = os.path.join(self.save_dir, 'result_bounding_boxes.npy')
        np.save(file_path, detected_bounding_boxes)


def main():
    parser = argparse.ArgumentParser(description='Test network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--save_dir', help='Path to save location')
    parser.add_argument('--representation', default="")
    parser.add_argument('--use_multiprocessing', help='If multiprocessing should be used', action='store_true')
    parser.add_argument('--compute_active_sites', help='If active sites should be calculated', action='store_true')

    args = parser.parse_args()
    settings_filepath = args.settings_file
    save_dir = args.save_dir
    
    settings = Settings(settings_filepath, generate_log=False)

    tester = TestSparseVGG(args, settings, save_dir)
    tester.test_sparse_VGG()


if __name__ == "__main__":
    main()
