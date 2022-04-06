#ifndef EVENT_PREPROCESSOR_H
#define EVENT_PREPROCESSOR_H

#include <ros/ros.h>

#include <iostream>
#include <stdio.h>
#include <math.h>

#include <std_msgs/Bool.h>
#include <dvs_msgs/EventArray.h>
#include <event_preprocessor/HistogramStamped.h>

#include <Eigen/Dense>
#include "opencv2/opencv.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h> 

#include <deque>

class EventPreprocessor{
  public:
    EventPreprocessor(int, char**, ros::NodeHandle& nh_input, ros::NodeHandle& nh_private_input);
    ~EventPreprocessor();

    void rawEventCallback(const dvs_msgs::EventArray& msg);
    void loopCallback(const ros::TimerEvent& event);

    void publishHistogramFromRawEventCallback(const dvs_msgs::EventArray& msg);
    void publishHistogramFromQueue(const std::deque<dvs_msgs::Event>& deque);

  private:

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    // ROS Timer
    ros::Timer loop_timer_;
    // ROS Subscriber
    ros::Subscriber raw_event_sub_; 
    // ROS Publisher
    ros::Publisher histogram_pub_;  
    // image_transport::Publisher histogram_image_pub_;


    bool is_sim_;
    std::deque<dvs_msgs::Event> event_queue_;
    double number_event_window_;

};

#endif /* EVENT_PREPROCESSOR_H */