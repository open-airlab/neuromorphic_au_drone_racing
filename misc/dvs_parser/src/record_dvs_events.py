#!/usr/bin/env python2

# @author Kristoffer Fogh Andersen <201510430@post.au.dk>
import os, sys
import rospy
import math
import numpy as np
import time
from datetime import datetime
from std_msgs.msg import Header
from dvs_msgs.msg import EventArray
from prophesee import dat_events_tools


class dvs_parser:
    def setup(self):

        #Set up publishers and subscribers
        rospy.init_node('dvs_recorder', anonymous=True, disable_signals=True)
        self.sub_state = rospy.Subscriber('dvs/events', EventArray,  self.event_callback)
        self.H = 180
        self.W = 240
        date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.event_filename = date + '_' + str(int(time.time())) + '_td.dat'
        self.event_file = dat_events_tools.write_header(self.event_filename, self.H, self.W)
        self.EVENT_TYPE = np.dtype(
            [("timestamp", "f8"), ("x", "u2"), ("y", "u2"), ("polarity", "b")], align=True)
        self.start_time = 0
        self.first = True


    def event_callback(self, data):
        if self.first == True:
            nsecs = (data.events[0].ts.secs) + (float(data.events[0].ts.nsecs)/1000000000)
            self.start_time = int(nsecs*1000000)
            self.first = False

        self.output_events = np.zeros((len(data.events)), dtype=self.EVENT_TYPE)
        for i in range(len(data.events)):
            nsecs = (data.events[i].ts.secs) + (float(data.events[i].ts.nsecs)/1000000000)
            self.output_events[i]['timestamp'] = int(nsecs*1000000) - self.start_time
            self.output_events[i]['x'] = data.events[i].y
            self.output_events[i]['y'] = data.events[i].x
            self.output_events[i]['polarity'] = data.events[i].polarity
        
        dat_events_tools.write_event_buffer(self.event_file, self.output_events)


def main():
    parser = dvs_parser()
    parser.setup()

    try:
        start = rospy.get_rostime()
        while (rospy.get_rostime() - start) / 1e9:
            time.sleep(1)
        parser.event_file.close()
        print("saving and exiting")
    except KeyboardInterrupt:
        parser.event_file.close()
        print("saving and exiting")

if __name__ == '__main__':
    main()
