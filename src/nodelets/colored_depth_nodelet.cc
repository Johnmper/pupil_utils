/**
@author     johnmper
@file       colored_depth_nodelet.cc
@brief      ROS nodelet that creates a colored depth image using a colormap.
@details    ROS nodelet that creates a colored depth image using a colormap. It subscribes to [pupil/depth/image_raw]
    whih is a MONO16 image, and creates a BGR8 color image with a colormap RED meaning NEAR and BLUE meaning FAR.
    Takes about 2 ms to run in the 640x480 depth image
*/


// C++ STANDARD INCLUDES
#include <iostream>
#include <string>
#include <vector>

// ROS INCLUDES
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

// ROS MESSAGES INCLUDES
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

// OPENCV INCLUDES
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

// LOCAL INCLUDES
#include <ros_utils/style.hh>

#undef CALC_MEAN
#undef SAVE_DEPTH

//===========================================================================================================
//      ######                            #     #
//      #     # #    # #####  # #         #     # ##### # #       ####
//      #     # #    # #    # # #         #     #   #   # #      #
//      ######  #    # #    # # #         #     #   #   # #       ####
//      #       #    # #####  # #         #     #   #   # #           #
//      #       #    # #      # #         #     #   #   # #      #    #
//      #        ####  #      # ######     #####    #   # ######  ####
//===========================================================================================================
namespace pupil_utils{


//===========================================================================================================
//    #####                                            ######
//   #     #  ####  #       ####  #####  ###### #####  #     # ###### #####  ##### #    #
//   #       #    # #      #    # #    # #      #    # #     # #      #    #   #   #    #
//   #       #    # #      #    # #    # #####  #    # #     # #####  #    #   #   ######
//   #       #    # #      #    # #####  #      #    # #     # #      #####    #   #    #
//   #     # #    # #      #    # #   #  #      #    # #     # #      #        #   #    #
//    #####   ####  ######  ####  #    # ###### #####  ######  ###### #        #   #    #
//===========================================================================================================
    class ColoredDepth: public nodelet::Nodelet{
    private:

        /// Depth subscriber
        image_transport::Subscriber depth_sub;

        /// Colored depth message
        sensor_msgs::Image colored_msg;
        /// Colored depth publisher
        image_transport::Publisher colored_pub;

        /// Minimal distance considered in the colored image in milimeter
        const uint16_t depth_min_z = 800U;
        /// Maximum distanve considered in the colored image in milimeter
        const uint16_t depth_max_z = 3500U;

        #ifdef SAVE_DEPTH
        cv::Mat depth_mean;
        cv::Mat depth_var;
        #endif

    public:
        ColoredDepth() {
            std::cout << LOG_ID("ColoredDepth") << "Created" << std::endl;
            #ifdef SAVE_DEPTH
            depth_var = cv::Mat::zeros(480,640,CV_16UC1);
            depth_mean = cv::Mat::zeros(480,640,CV_16UC1);
            #endif
        }

        virtual ~ColoredDepth(){
        }

    private:
        virtual void onInit(){

            ros::NodeHandle nh = getNodeHandle();
            image_transport::ImageTransport it(nh);

            colored_msg.header.seq = 0;
            colored_msg.header.frame_id = "rgb_optical_frame";
            colored_pub = it.advertise("/pupil/depth/image_colored", 1);

            image_transport::TransportHints depth_hints("raw", ros::TransportHints(), getPrivateNodeHandle(), "depth_image_transport");
            depth_sub = it.subscribe("/pupil/depth/image_raw",1,&pupil_utils::ColoredDepth::depthCallback,this,depth_hints);

            std::cout << LOG_ID("ColoredDepth") << "Initialization Complete" << std::endl;
        }

        /// Depth image callback
        void depthCallback(const sensor_msgs::ImageConstPtr& depth_msg){
            // Update sequence
            ++colored_msg.header.seq;

            // Check if any node is currently needing the colored depth image if not then dont publish
            if(colored_pub.getNumSubscribers() != 0 ){

                // Populate colored_msg data
                colored_msg.height = depth_msg->height;
                colored_msg.width = depth_msg->width;
                colored_msg.step = 3*depth_msg->width;
                colored_msg.encoding = sensor_msgs::image_encodings::BGR8;
                colored_msg.data.resize(3*depth_msg->height*depth_msg->width);
                auto it = colored_msg.data.begin();
                uint32_t sz = depth_msg->height*depth_msg->width;
                uint16_t* T = ((uint16_t*)depth_msg->data.data());
                for(uint32_t nn=0; nn < sz; ++nn){
                    if( T[nn] > 3500U || T[nn] < 800U ){
                        // More than 3.5m or less than 0.8m
                        *it++=0;    // Blue
                        *it++=0;    // Green
                        *it++=0;    // Red
                    }
                    else{
                        *it++ = (uint8_t)((T[nn]-800U)/11);      // Blue
                        *it++ = 0;                               // Green
                        *it++ = (uint8_t)(255U-(T[nn]-800U)/11); // Red
                    }
                }


                #ifdef CALC_MEAN
                #ifdef SAVE_DEPTH
                depth_mean = cv::Mat::zeros(480,640,CV_16UC1);
                #endif
                double tot = 0;
                double nn = 0;
                uint32_t mid_i = depth_msg->height/2;
                uint32_t mid_j = depth_msg->width/2;
                uint16_t* D = ((uint16_t*)depth_msg->data.data());
                for(uint32_t ii=0; ii < depth_msg->height; ++ii){
                    uint16_t* R = &D[ii*depth_msg->width];
                    for(uint32_t jj=0; jj < depth_msg->width; ++jj){
                        if(    R[jj] < 3500U && R[jj] > 710U)
                        {
                            #ifdef SAVE_DEPTH
                            depth_mean.at<uint16_t>(ii,jj) = R[jj];
                            #endif
                            tot += R[jj];
                            nn += 1.0;
                        }
                    }
                }
                tot /= nn;
                std::cout << "Seq: " <<  colored_msg.header.seq << " Mean: " << tot << std::endl;
                #endif

                #ifdef SAVE_DEPTH
                if(colored_msg.header.seq < 10){
                    cv::imwrite( std::string(std::string("/home/johnmper/Thesis/Data/r200_depth/a60_2/seq_0")+std::to_string(colored_msg.header.seq-1)+std::string(".png")).c_str(), depth_mean );
                }
                else if(colored_msg.header.seq < 100){
                    cv::imwrite( std::string(std::string("/home/johnmper/Thesis/Data/r200_depth/a60_1/seq_")+std::to_string(colored_msg.header.seq-1)+std::string(".png")).c_str(), depth_mean );
                }
                #endif
                colored_msg.header.stamp = ros::Time::now();
                colored_pub.publish(colored_msg);
            }

        }

    };

    PLUGINLIB_DECLARE_CLASS(pupil_utils, ColoredDepth, pupil_utils::ColoredDepth, nodelet::Nodelet);

}
