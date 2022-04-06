#include "event_preprocessor/event_preprocessor.h"


// Constructor
EventPreprocessor::EventPreprocessor(int argc, char** argv, ros::NodeHandle& nh_input, ros::NodeHandle& nh_private_input)
    : nh_(nh_input), nh_private_(nh_private_input)
{
  // Subscribing & Publishing
  loop_timer_ = nh_.createTimer(ros::Duration(0.1), &EventPreprocessor::loopCallback, this); // loop rate: 4 HZ
  raw_event_sub_ = nh_.subscribe("/dvs/events", 1, &EventPreprocessor::rawEventCallback, this,ros::TransportHints().tcpNoDelay());
  histogram_pub_ = nh_.advertise<event_preprocessor::HistogramStamped>("/event_preprocessor/event_histogram", 10);

  // image_transport::ImageTransport *it_ = new image_transport::ImageTransport(nh_);
  // histogram_image_pub_ = it_->advertise("/event_preprocessor/event_histogram", 1);

  nh_private_.param<bool>("is_sim", is_sim_, true);
  nh_private_.param<double>("number_event_window", number_event_window_, 10000);

}

// Destructor
EventPreprocessor::~EventPreprocessor(){
  ros::shutdown();
  exit(0);
}


void EventPreprocessor::publishHistogramFromRawEventCallback(const dvs_msgs::EventArray& msg) {
  
  event_preprocessor::HistogramStamped histogram_msg;

  for (int i = 0; i < msg.height * msg.width; i ++){
    histogram_msg.positive.push_back(0);
    histogram_msg.negative.push_back(0);
  }

  for (int i = 0; i < msg.events.size(); i ++){
    
    if (msg.events[i].polarity == true){
      histogram_msg.positive[msg.events[i].x + msg.width*msg.events[i].y] ++;     
    }
    else {
      histogram_msg.negative[msg.events[i].x + msg.width*msg.events[i].y] ++;     
    }
  }

  histogram_msg.header.stamp = ros::Time::now();

  histogram_pub_.publish(histogram_msg);

  std::cout << "data events size = " << msg.events.size() << std::endl;
}


void EventPreprocessor::publishHistogramFromQueue(const std::deque<dvs_msgs::Event>& deque) {

}



void EventPreprocessor::rawEventCallback(const dvs_msgs::EventArray& msg) {
  int start_s=clock();

  // publishHistogramFromRawEventCallback(); // for raw

  // Place them into a deque
  for (int i = 0; i < msg.events.size(); i ++){
    
    if (event_queue_.size() < number_event_window_ ){
      event_queue_.push_back(msg.events[i]);
    }
    else{
      event_queue_.pop_front();
        event_queue_.push_back(msg.events[i]);
    }
  }


  event_preprocessor::HistogramStamped histogram_msg;

  for (int i = 0; i < msg.height * msg.width; i ++){
    histogram_msg.positive.push_back(0);
    histogram_msg.negative.push_back(0);
  }

  for (int i = 0; i < event_queue_.size(); i ++){
    
    if (event_queue_[i].polarity == true){
      histogram_msg.positive[event_queue_[i].x + msg.width*event_queue_[i].y] ++;     
    }
    else {
      histogram_msg.negative[event_queue_[i].x + msg.width*event_queue_[i].y] ++;     
    }
  }
  
  histogram_msg.header.stamp = ros::Time::now();

  histogram_pub_.publish(histogram_msg);

  std::cout << "single message data events size = " << msg.events.size() << std::endl;

  int stop_s=clock();
  std::cout << "[Event_preprocessor] total time = " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << " ms" << std::endl;

}


void EventPreprocessor::loopCallback(const ros::TimerEvent& event){

}
