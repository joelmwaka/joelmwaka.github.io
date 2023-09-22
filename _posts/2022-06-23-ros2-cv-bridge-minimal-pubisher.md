---
layout: post
title:  "ROS2 CV_Bridge Example in Cpp and Python"
date:   2022-06-23 10:00:00 +0000
categories: ROS2 cv_bridge cpp python
---

In this post, I show how to use cv_bridge in a minimal frames publisher node in ROS2 written in C++. Python code is also included for the relevant part below.

A short usage example. For a full node example, see below.

```cpp

cv::Mat frame;
cap.read(frame);

//create ROS2 messages
sensor_msgs::msg::Image _img_msg;
std_msgs::msg::Header _header;
cv_bridge::CvImage _cv_bridge;
_header.stamp = this->get_clock() -> now();
_cv_bridge = cv_bridge::CvImage(_header, sensor_msgs::image_encodings::BGR8, frame);
_cv_bridge.toImageMsg(_img_msg);

//publish
_image_publisher_ -> publish(_img_msg);

```

A full node example.

```cpp

#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std::chrono_literals;

class CameraPublisher: public rclcpp::Node
{

    public:
        int cam_id;
        cv::VideoCapture cap;

        //constructor
        CameraPublisher():Node("camera_publisher")
        {

            //parameters
            this -> declare_parameter("cam_id", 4); //device_id for webcam
            this -> declare_parameter("camera_img_topic", "/camera/color/image_raw"); //frame publish topic name

            //open the video stream
            this -> get_parameter("cam_id", cam_id);
            cap = open_stream(cam_id);

            //create the image publisher and timer
            std::string camera_publish_topic_name;
            this -> get_parameter("camera_img_topic", camera_publish_topic_name);
            _image_publisher_ = this -> create_publisher<sensor_msgs::msg::Image>(camera_publish_topic_name, 1);
            _image_timer_ = this -> create_wall_timer(0.03s, std::bind(&CameraPublisher::publish_frame, this));

        }

        //Opens the camera stream and sets to high resolution.
        cv::VideoCapture open_stream(int cam_id)
        {
            cap.open(cam_id, cv::CAP_V4L2); //CAP_V4L2 is linux only.
            if (!cap.isOpened())
            {   
                RCLCPP_ERROR(this->get_logger(), "Camera could not be opened on device id: '%i'", cam_id);
                exit(0);
            }

            RCLCPP_INFO(this->get_logger(), "Camera opened on device id: '%i'", cam_id);

            //set resolution
            cap.set(3, 1920);
            cap.set(4, 1080);

            return cap;
        }


    private:
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _image_publisher_;
        rclcpp::TimerBase::SharedPtr _image_timer_;

        void publish_frame()
        {
            //read camera frame
            cv::Mat frame;
            cap.read(frame);
            if (frame.empty()){
                RCLCPP_WARN(this->get_logger(), "Frame data is emtpy");
                return;
            }

            //create ROS2 messages
            sensor_msgs::msg::Image _img_msg;
            std_msgs::msg::Header _header;
            cv_bridge::CvImage _cv_bridge;
            _header.stamp = this->get_clock() -> now();
            _cv_bridge = cv_bridge::CvImage(_header, sensor_msgs::image_encodings::BGR8, frame);
            _cv_bridge.toImageMsg(_img_msg);

            //publish
            _image_publisher_ -> publish(_img_msg);
        }


};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraPublisher>());
  rclcpp::shutdown();
  return 0;
}

```


Next add the required packages to ```CMakeLists.txt```. In this case, my program name is ```camera_publisher_cpp```, which may be different for you.

```cmake
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

add_executable(camera_publisher_cpp src/camera_publisher.cpp)
ament_target_dependencies(camera_publisher_cpp rclcpp std_msgs sensor_msgs OpenCV cv_bridge)

install(TARGETS camera_publisher_cpp DESTINATION lib/${PROJECT_NAME})

```

Finally, update the ```packages.xml``` file to add dependencies.

```xml
  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <depend>OpenCV</depend>
  <depend>sensor_msgs</depend>
  <depend>cv_bridge</depend>
```

Use ```colcon build ``` to build your package.

**Python**

For python, only the cv_bridge part is shown below.

```python
...
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
...

class FramesPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        ...
        self.bridge = CvBridge()
        ...
        

    def publish_frames(self):
        ret, frame = self.vid.read() #opencv returned video frame.
        
        #convert to ROS2 Image msg.
        img = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        img.header.stamp = self.get_clock().now().to_msg()
        img.header.frame_id = self.camera_name        
        
        #publish
        self.img_publisher.publish(img)
```

Update the ```package.xml``` file. 
```xml
  ...
  <exec_depend>cv_bridge</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  ...
```

use ```colcon build``` to build your package.
