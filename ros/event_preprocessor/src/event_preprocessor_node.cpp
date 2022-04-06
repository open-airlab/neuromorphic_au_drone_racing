#include "event_preprocessor/event_preprocessor.h"

//using namespace RAI;
int main(int argc, char** argv) {
  ros::init (argc, argv, "Event_preprocess_node");
  ros::NodeHandle nh("");
  ros::NodeHandle nh_private("~");

  EventPreprocessor* event_preprocessor = new EventPreprocessor(argc, argv, nh, nh_private);

  // dynamic_reconfigure::Server<geometric_controller::GeometricControllerConfig> srv;
  // dynamic_reconfigure::Server<geometric_controller::GeometricControllerConfig>::CallbackType f;
  // f = boost::bind(&geometricCtrl::dynamicReconfigureCallback, geometricController, _1, _2);
  // srv.setCallback(f);

  ros::spin();
  return 0;
}