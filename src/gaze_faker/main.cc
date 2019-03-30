// STANDARD C++ INCLUDES
#include<iostream>
#include<memory>

// ROS INCLUDES
#include<ros/ros.h>
#include<pupil_msgs/gaze_datum.h>
#include<pupil_msgs/eye_status.h>
#include<cv_bridge/cv_bridge.h>
#include<image_transport/image_transport.h>
#include<sensor_msgs/Image.h>
#include<sensor_msgs/CameraInfo.h>
#include<sensor_msgs/image_encodings.h>

// OTHER INCLUDES
#include<Eigen/Eigen> // Full eigen library

// OPENCV INCLUDES
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

// LOCAL INCLUDES
#include<ros_utils/style.hh>

#define WINDOW_NAME_RGB "Gaze Faker"
#define NODE_NAME       "gaze_faker"
#define BLINK_TIME_MEAN 3.5
#define BLINK_TIME_VAR  1.0
#define GAZE_MEAN       0.0
#define GAZE_VAR        0.0

volatile int num_blinks = 0;
volatile double last_blink_time = 0;
volatile bool fake_blink = false;
volatile bool fake_gaze = false;
volatile int x_lst = 0;
volatile int y_lst = 0;


//===========================================================================================================
//   ######                         #     #
//   #     # #      # #    # #    # ##    #  ####  #    #
//   #     # #      # ##   # #   #  # #   # #    # #    #
//   ######  #      # # #  # ####   #  #  # #    # #    #
//   #     # #      # #  # # #  #   #   # # #    # # ## #
//   #     # #      # #   ## #   #  #    ## #    # ##  ##
//   ######  ###### # #    # #    # #     #  ####  #    #
//===========================================================================================================
void blinkNow(){
    if(fake_gaze){
        last_blink_time = ros::Time::now().toSec();
        std::cout << style::green << style::bold << " BLINK " << style::normal << last_blink_time << " (" << x_lst <<", "<<y_lst<<")"<< std::endl;
        fake_blink = true;
    }
}

//===========================================================================================================
//                                       #####
//   #    #  ####  #    #  ####  ###### #     #   ##   #      #      #####    ##    ####  #    #
//   ##  ## #    # #    # #      #      #        #  #  #      #      #    #  #  #  #    # #   #
//   # ## # #    # #    #  ####  #####  #       #    # #      #      #####  #    # #      ####
//   #    # #    # #    #      # #      #       ###### #      #      #    # ###### #      #  #
//   #    # #    # #    # #    # #      #     # #    # #      #      #    # #    # #    # #   #
//   #    #  ####   ####   ####  ######  #####  #    # ###### ###### #####  #    #  ####  #    #
//===========================================================================================================
void mouse_callback(int event, int  x, int  y, int flag, void *param){

    if(event == cv::EVENT_MOUSEMOVE && fake_gaze){
        x_lst = x;
        y_lst = y;
    }

    if(event == cv::EVENT_LBUTTONDOWN && fake_gaze){
        x_lst = x;
        y_lst = y;
        blinkNow();
        ++num_blinks;
    }

}

//===========================================================================================================
//    #####                       #######
//   #     #   ##   ###### ###### #         ##   #    # ###### #####
//   #        #  #      #  #      #        #  #  #   #  #      #    #
//   #  #### #    #    #   #####  #####   #    # ####   #####  #    #
//   #     # ######   #    #      #       ###### #  #   #      #####
//   #     # #    #  #     #      #       #    # #   #  #      #   #
//    #####  #    # ###### ###### #       #    # #    # ###### #    #
//===========================================================================================================
class GazeFaker{
private:

    /// Ros nodehandle
    ros::NodeHandle node;
    /// Image Transport object
    image_transport::ImageTransport it;
    /// Image Transport subscriber
    image_transport::Subscriber rgb_sub;
    /// Fake gaze message publisher
    ros::Publisher gaze_pub;
    /// Fake eye status message publisher
    ros::Publisher eye_status_pub;
    /// Fake gaze message
    pupil_msgs::gaze_datum gaze_msg;
    /// Fake eye_status message
    pupil_msgs::eye_status eye_status_msg;

    /// Timer to force blinking at each 4 seconds
    ros::Timer blink_timer;
    /// Empty image for blink
    cv::Mat blink_image;

    /// WORLD intrinsics focal length X
    double fx;
    /// WORLD intrinsics focal length Y
    double fy;
    /// WORLD intrinsics image center X
    double cx;
    /// WORLD intrinsics image center Y
    double cy;

    /// Necessary object for the random generators
    std::random_device rd;
    /// Necessary object for the random generators
    std::mt19937 gen;
    /// Normal distribution generator
    std::normal_distribution<double> d;
    /// Normal Distribution generator for the Gaze in pixel
    std::normal_distribution<double> d_gaze;


//===========================================================================================================
//    #####
//   #     #  ####  #    #  ####  ##### #####  #    #  ####  #####  ####  #####
//   #       #    # ##   # #        #   #    # #    # #    #   #   #    # #    #
//   #       #    # # #  #  ####    #   #    # #    # #        #   #    # #    #
//   #       #    # #  # #      #   #   #####  #    # #        #   #    # #####
//   #     # #    # #   ## #    #   #   #   #  #    # #    #   #   #    # #   #
//    #####   ####  #    #  ####    #   #    #  ####   ####    #    ####  #    #
//===========================================================================================================
public:
    GazeFaker() : it(node), rd(), gen(rd()), d(BLINK_TIME_MEAN,BLINK_TIME_VAR), d_gaze(GAZE_MEAN,GAZE_VAR){
        // Initialization of blink time
        last_blink_time = ros::Time::now().toSec();
        blink_image = cv::Mat::zeros(1080,1920,CV_8UC3);

        // Create OpenCv Window
        cv::namedWindow(WINDOW_NAME_RGB, CV_GUI_NORMAL | cv::WINDOW_NORMAL);
        cv::resizeWindow(WINDOW_NAME_RGB, 960, 540);
        cv::moveWindow(WINDOW_NAME_RGB, 50, 50); // resolves a bug that places the window partial outside screen
        // Mouse callback for faking gaze point
        cv::setMouseCallback(WINDOW_NAME_RGB, mouse_callback);

        // Subscribe to the world stream of r200 simulated camera
        rgb_sub = it.subscribe("/pupil/world/image_raw", 1, &GazeFaker::rgbCallback, this);

        // Publisher for the gaze
        gaze_pub = node.advertise<pupil_msgs::gaze_datum>("/pupil/gaze_0/datum",1);
        eye_status_pub = node.advertise<pupil_msgs::eye_status>("/pupil/gaze_0/eye_status",1);
        // Populate gaze_datum static data
        gaze_msg.topic = "gaze.3d.0";
        gaze_msg.header.frame_id = "r200_camera_link";


        // One Time
        double next_blink = d(gen);
        while(next_blink < 0){  // Check if next_blink is negative and retry if that is the case
            next_blink = d(gen);
        }
        //std::cout << "Next blink in "<< next_blink << std::endl;
        blink_timer = node.createTimer(ros::Duration(next_blink), &GazeFaker::checkBlink, this, true);

        // Message to store the camera info for the world camera (oneshot, it is only populated here
        sensor_msgs::CameraInfo rgb_info;
        rgb_info = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/pupil/world/camera_info", ros::Duration(10));
        fx =  rgb_info.K[0];
        fy =  rgb_info.K[4];
        cx =  rgb_info.K[2];
        cy =  rgb_info.K[5];

        std::cout << "Using intrinsics: " << std::endl;
        std::cout << " fx: " << fx << std::endl;
        std::cout << " fy: " << fy << std::endl;
        std::cout << " cx: " << cx << std::endl;
        std::cout << " cy: " << cy << std::endl;

    }
    ~GazeFaker(){
        cv::destroyAllWindows();
    }


//===========================================================================================================
//   ######                 #####
//   #     #  ####  #####  #     #   ##   #      #      #####    ##    ####  #    #
//   #     # #    # #    # #        #  #  #      #      #    #  #  #  #    # #   #
//   ######  #      #####  #       #    # #      #      #####  #    # #      ####
//   #   #   #  ### #    # #       ###### #      #      #    # ###### #      #  #
//   #    #  #    # #    # #     # #    # #      #      #    # #    # #    # #   #
//   #     #  ####  #####   #####  #    # ###### ###### #####  #    #  ####  #    #
//===========================================================================================================
private:
    void rgbCallback(const sensor_msgs::ImageConstPtr& msg){
        static cv_bridge::CvImagePtr rgb_ptr;
        try{
            // No copy made, img_ptr->image to access msg image data directly
            rgb_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch(cv_bridge::Exception& e ){
            ROS_ERROR("CV_BRIDGE exception: %s", e.what());
            return;
        }

        if(fake_gaze){
            if(x_lst < 18){
                x_lst = 18;
            }else if(x_lst > msg->width-18){
                x_lst = msg->width-18;
            }
            if(y_lst < 18){
                y_lst = 18;
            }else if(y_lst > msg->height-18){
                y_lst = msg->height-18;
            }

            produceGaze(msg);
        }

        if(fake_blink && fake_gaze){
            fake_blink = false;
            cv::imshow(WINDOW_NAME_RGB,blink_image);
        }
        else{
            cv::circle(rgb_ptr->image, cv::Point(x_lst,y_lst), 18, cv::Scalar(40,40,180), CV_FILLED);
            cv::imshow(WINDOW_NAME_RGB,rgb_ptr->image);
        }

        if(cv::waitKey(3) == 'g'){
            // Start/Stop Fake gaze from mouse position
            if(fake_gaze){
                fake_gaze = false;
                std::cout << style::magenta << "Fake gaze OFF" << style::normal << std::endl;
            }else{
                fake_gaze = true;
                std::cout << style::magenta << "Fake gaze ON" << style::normal << std::endl;
            }
        }
    }

//===========================================================================================================
//    #####                              ######
//   #     # #    # ######  ####  #    # #     # #      # #    # #    #
//   #       #    # #      #    # #   #  #     # #      # ##   # #   #
//   #       ###### #####  #      ####   ######  #      # # #  # ####
//   #       #    # #      #      #  #   #     # #      # #  # # #  #
//   #     # #    # #      #    # #   #  #     # #      # #   ## #   #
//    #####  #    # ######  ####  #    # ######  ###### # #    # #    #
//===========================================================================================================
    void checkBlink(const ros::TimerEvent& event){
        if(num_blinks == 0){

            // TODO: BLINK NOW!
            blinkNow();

            double next_blink = d(gen);
            while(next_blink < 0){  // Check if next_blink is negative and retry if that is the case
                next_blink = d(gen);
            }

            // Recall checkBlink in random distribution time from now
            //std::cout << "  Next blink in "<< next_blink << std::endl;
            blink_timer = node.createTimer(ros::Duration(next_blink), &GazeFaker::checkBlink, this, true);
        }
        else{
            //std::cout << "  Fake blink occurred before" << std::endl;

            double next_blink = d(gen);
            while(next_blink < 0){  // Check if next_blink is negative and retry if that is the case
                next_blink = d(gen);
            }

            if(next_blink-(event.current_real.toSec()-last_blink_time) < 0){
                // TODO: BLINK NOW!
                blinkNow();

                // Recalculate the blink time
                next_blink = d(gen);
                while(next_blink < 0){  // Check if next_blink is negative and retry if that is the case
                    next_blink = d(gen);
                }
            }
            else{
                next_blink = next_blink-(event.current_real.toSec()-last_blink_time);
            }

            // Recall checkBlink in random distribution time from last blink time
            //std::cout << "  Next blink in "<< next_blink << std::endl;
            blink_timer = node.createTimer(ros::Duration(next_blink), &GazeFaker::checkBlink, this, true);
        }

        num_blinks = 0;
    }

//===========================================================================================================
//   ######                                             #####
//   #     # #####   ####  #####  #    #  ####  ###### #     #   ##   ###### ######
//   #     # #    # #    # #    # #    # #    # #      #        #  #      #  #
//   ######  #    # #    # #    # #    # #      #####  #  #### #    #    #   #####
//   #       #####  #    # #    # #    # #      #      #     # ######   #    #
//   #       #   #  #    # #    # #    # #    # #      #     # #    #  #     #
//   #       #    #  ####  #####   ####   ####  ######  #####  #    # ###### ######
//===========================================================================================================
    void produceGaze(const sensor_msgs::ImageConstPtr& msg){
        ++gaze_msg.header.seq;
        gaze_msg.header.stamp = msg->header.stamp;
        gaze_msg.timestamp = gaze_msg.header.stamp.toSec();
        gaze_msg.norm_pos.x = (x_lst+d_gaze(gen)-cx)/fx;
        gaze_msg.norm_pos.y = (y_lst+d_gaze(gen)-cy)/fy;
        // double x = (gaze_msg.timestamp-last_blink_time);
        if(fake_blink){
            gaze_msg.confidence = 0;
        }
        else{
            gaze_msg.confidence = 1.0; // 1.0-1.0/(1.0+100.0*x*x);  // Simulated Confidence depends on the amount of time since last blink
        }
        // Simulated Eye center is centered in the camera axis
        gaze_msg.eye_center_3d.x = 0;
        gaze_msg.eye_center_3d.y = 0;
        gaze_msg.eye_center_3d.z = 0;
        // Simulated Gaze normal passes through the Camera Axis and the respective pixel in the image plane!
        gaze_msg.gaze_normal_3d.x = gaze_msg.norm_pos.x;
        gaze_msg.gaze_normal_3d.y = gaze_msg.norm_pos.y;
        gaze_msg.gaze_normal_3d.z = 1;

        // Publish the artificial gaze
        gaze_pub.publish(gaze_msg);

        // std_msgs/Header header
        // bool is_closed
        // bool is_fixating
        // bool blink
        // bool double_blink
        // float64 eye_shade
        // float64 fixation_dispersion
        // float64 fixation_confidence
        // # Following parameters come directly from gaze_datum message
        // float64 confidence
        // point3d gaze_normal_3d
        // point3d eye_center_3d

        eye_status_msg.header = msg->header;
        eye_status_msg.is_closed = fake_blink;
        eye_status_msg.is_fixating = true;  // TODO: use norm_pos.x to calculate the fixation
        eye_status_msg.blink = false;
        eye_status_msg.double_blink = fake_blink;    // TODO:
        eye_status_msg.eye_shade = 0;
        eye_status_msg.fixation_dispersion = 0;
        eye_status_msg.fixation_confidence = 1;
        eye_status_msg.confidence = 1;
        eye_status_msg.gaze_normal_3d = gaze_msg.gaze_normal_3d;
        eye_status_msg.eye_center_3d = gaze_msg.eye_center_3d;
        //
        eye_status_pub.publish(eye_status_msg);
    }
};


//===========================================================================================================
//      #     #
//      ##   ##   ##   # #    #
//      # # # #  #  #  # ##   #
//      #  #  # #    # # # #  #
//      #     # ###### # #  # #
//      #     # #    # # #   ##
//      #     # #    # # #    #
//===========================================================================================================
int main(int argc, char** argv){

    ros::init(argc,argv,NODE_NAME);

    GazeFaker gaze_faker;
    ros::spin();

    return 0;
}
