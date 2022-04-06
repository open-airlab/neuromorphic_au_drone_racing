"""
small executable to show the content of the Prophesee dataset
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import cv2
import argparse
from glob import glob

from src.visualize import vis_utils as vis

from src.io.psee_loader import PSEELoader


def play_files_parallel_withbox(td_files, labels=None, delta_t=80000, skip=0):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    # open the video object for the input files
    videos = [PSEELoader(td_file) for td_file in td_files]
    # use the naming pattern to find the corresponding box file
    box_videos1 = [PSEELoader(glob(td_file.split('_td.dat')[0] +  '*_bbox.npy')[0]) for td_file in td_files]
    #box_videos2 = [PSEELoader(glob(td_file.split('_td.dat')[0] +  '*_resultRNN.npy')[0]) for td_file in td_files]
    
    height, width = videos[0].get_size()
    if height == 180:
        labelmap = vis.GATE_LABELMAP
    elif height == 240:
        labelmap = vis.LABELMAP 
    else:
        labelmap = vis.LABELMAP_LARGE

    # optionally skip n minutes in all videos
    for v in videos + box_videos1:
        v.seek_time(skip)

    # preallocate a grid to display the images
    size_x = int(math.ceil(math.sqrt(len(videos))))
    size_y = int(math.ceil(len(videos) / size_x))
    frame = np.zeros((size_y * height, width * size_x, 3), dtype=np.uint8)

    cv2.namedWindow('Ground Truth', cv2.WINDOW_NORMAL)
    last_boxes = np.empty((0,))
    ts = 0
    ts_last_box = 0
    lol = input("Press any key to start ground truth visualization")
    counter = 0

    print("Visualizing ground truth")
    # while all videos have something to read
    """
    while not sum([video.done for video in videos]):

        # load events and boxes from all files
        events = [video.load_n_events(5000) for video in videos]
        box_events = [box_video.load_delta_t(delta_t) for box_video in box_videos1]
        for index, (evs, boxes) in enumerate(zip(events, box_events)):
            y, x = divmod(index, size_x)

            if len(evs) != 0:
                ts = evs[0][0]
            # put the visualization at the right spot in the grid
            im = frame[y * height:(y + 1) * height, x * width: (x + 1) * width]
            # call the visualization functions
            im = vis.make_binary_histo(evs, img=im, width=width, height=height)
            vis.draw_bboxes(im, boxes, labelmap=labelmap)

            if len(boxes) == 0:
                vis.draw_bboxes(im, last_boxes, labelmap=labelmap)
            else:
                last_boxes = boxes.copy()
                ts_last_box = ts
            if ts - ts_last_box > 200000:
                last_boxes = np.empty((0,))
        key = input("step")
        counter += 1


        # display the result
        cv2.imshow('Ground Truth', frame)
        cv2.waitKey(1)


    """
    while not sum([video.done for video in videos]):

        # load events and boxes from all files
        events = [video.load_delta_t(delta_t) for video in videos]
        box_events = [box_video.load_delta_t(delta_t) for box_video in box_videos1]
        for index, (evs, boxes) in enumerate(zip(events, box_events)):
            y, x = divmod(index, size_x)

            if len(evs) != 0:
                ts = evs[0][0]
            # put the visualization at the right spot in the grid
            im = frame[y * height:(y + 1) * height, x * width: (x + 1) * width]
            # call the visualization functions
            im = vis.make_binary_histo(evs, img=im, width=width, height=height)
            vis.draw_bboxes(im, boxes, labelmap=labelmap)

            if len(boxes) == 0:
                pass
                vis.draw_bboxes(im, last_boxes, labelmap=labelmap)
            else:
                last_boxes = boxes.copy()
                ts_last_box = ts
            if ts - ts_last_box > 50000:
                last_boxes = np.empty((0,))


        # display the result
        #key = input("lol")
        cv2.imshow('Ground Truth', frame)
        cv2.waitKey(1)


    videos = [PSEELoader(td_file) for td_file in td_files]
    cv2.namedWindow('Detected', cv2.WINDOW_NORMAL)
    last_boxes = np.empty((0,))
    ts = 0
    ts_last_box = 0
    lol = input("Press any key to start detected objects visualization")

    print("Visualizing detected boxes")

    # while all videos have something to read
    while not sum([video.done for video in videos]):

        # load events and boxes from all files
        events = [video.load_delta_t(delta_t) for video in videos]
        box_events = [box_video.load_n_events(10000) for box_video in box_videos2]
        for index, (evs, boxes) in enumerate(zip(events, box_events)):
            y, x = divmod(index, size_x)

            if len(evs) != 0:
                ts = evs[0][0]
            # put the visualization at the right spot in the grid
            im = frame[y * height:(y + 1) * height, x * width: (x + 1) * width]
            # call the visualization functions
            im = vis.make_binary_histo(evs, img=im, width=width, height=height)
            vis.draw_bboxes(im, boxes, labelmap=labelmap)

            if len(boxes) == 0:
                pass
                #vis.draw_bboxes(im, last_boxes, labelmap=labelmap)
            else:
                last_boxes = boxes.copy()
                ts_last_box = ts
            if ts - ts_last_box > 200000:
                last_boxes = np.empty((0,))


        # display the result
        cv2.imshow('Detected', frame)
        cv2.waitKey(1)


def play_files_parallel_withoutbox(td_files, labels=None, delta_t=80000, skip=0):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    # open the video object for the input files
    videos = [PSEELoader(td_file) for td_file in td_files]
    
    height, width = videos[0].get_size()
    labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE

    # preallocate a grid to display the images
    size_x = int(math.ceil(math.sqrt(len(videos))))
    size_y = int(math.ceil(len(videos) / size_x))
    frame = np.zeros((size_y * height, width * size_x, 3), dtype=np.uint8)

    cv2.namedWindow('SimEvents', cv2.WINDOW_NORMAL)
    print(videos[0].event_count())
    lol = input("Press any key to start visualization")

    # while all videos have something to read
    while not sum([video.done for video in videos]):

        # load events and boxes from all files
        events = [video.load_delta_t(delta_t) for video in videos]

        for index, (evs) in enumerate(events):
            y, x = divmod(index, size_x)

            if len(evs) != 0:
                ts = evs[0][0]
            # put the visualization at the right spot in the grid
            im = frame[y * height:(y + 1) * height, x * width: (x + 1) * width]
            # call the visualization functions
            im = vis.make_binary_histo(evs, img=im, width=width, height=height)

        # display the result
        cv2.imshow('SimEvents', frame)
        cv2.waitKey(1)
        key = input("lol")


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('records', nargs="+",
                        help='input event files, annotation files are expected to be in the same folder')
    parser.add_argument('-s', '--skip', default=0, type=int, help="skip the first n microseconds")
    parser.add_argument('-d', '--delta_t', default=15000, type=int, help="load files by delta_t in microseconds")
    parser.add_argument('-b', '--box', default=1, type=int, help="Whether to include bounding box file")

    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    withbox = ARGS.box
    if withbox == 1:
        print("withbox")
        play_files_parallel_withbox(ARGS.records, skip=ARGS.skip, delta_t=ARGS.delta_t)
    else:
        print("withoutbox")
        play_files_parallel_withoutbox(ARGS.records, skip=ARGS.skip, delta_t=ARGS.delta_t)
