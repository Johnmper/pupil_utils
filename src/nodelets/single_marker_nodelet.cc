/**
@author     johnmper
@file       p2p_nodelet.cc
@brief      ROS nodelet that calculates the transform between the JACO robot and R200 camera.
@details    ROS nodelet that calculates the transform between the JACO robot and R200 camera. Subscribes to
    [pupil/world/image_raw] uses fiducial markers in the JACO robot and P2P algorithm to estimate the camera frame.
*/

// C++ STANDARD INCLUDES
#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <utility>
#include <map>
#include <functional> // std::bind
#include <algorithm>
#include <chrono>

// ROS INCLUDES
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <cv_bridge/cv_bridge.h>

// ROS MESSAGES INCLUDES
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

// OPENCV INCLUDES
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Eigen>

#include <ros_utils/math.hh>
#include <ros_utils/style.hh>


using hr_clock = std::chrono::high_resolution_clock;
#define MS_CAST(a,b) std::chrono::duration_cast<std::chrono::microseconds>(a-b)

// Compilation Options
#define TIME_SINGLE
#define SHOW_WINDOW
#define SINGLE_POSE_ESTIMATION

namespace pupil_utils{
    struct PnPKstPose{
        std::string filepath;
        std::ofstream file;

        PnPKstPose(std::string filepath_) : filepath(filepath_){
            file.open(filepath,std::ofstream::out);
            if(file.fail()){
                std::cerr << "Failed to create " << filepath << " file." << std::endl;
            }else{
                file << "time, detect_duration, process_duration, tx, ty, tz, qx, qy, qz, qw";
                file << std::endl;
            }
        }
        ~PnPKstPose(){
            file.close();
        }

        bool add(double tm, unsigned int detect_duration, unsigned int process_duration, geometry_msgs::Transform pose_ ){

            if(!file.is_open()){
                std::cerr << "File isn't opened or was prematurely closed" << std::endl;
                return false;
            }

            file << tm << ", ";
            file << detect_duration << ", ";
            file << process_duration << ", ";
            file << pose_.translation.x << ", " << pose_.translation.y << ", " << pose_.translation.z << ", ";
            file << pose_.rotation.x << ", " << pose_.rotation.y << ", " << pose_.rotation.z << ", " << pose_.rotation.w;
            file << std::endl;

            return true;
        }
    };
//===========================================================================================================
//   ######   #####  ######  #     #
//   #     # #     # #     # ##    #  ####  #####  ###### #      ###### #####
//   #     #       # #     # # #   # #    # #    # #      #      #        #
//   ######   #####  ######  #  #  # #    # #    # #####  #      #####    #
//   #       #       #       #   # # #    # #    # #      #      #        #
//   #       #       #       #    ## #    # #    # #      #      #        #
//   #       ####### #       #     #  ####  #####  ###### ###### ######   #
//===========================================================================================================
    class SingleMarker: public nodelet::Nodelet{
    private:




        double start_time_;
        /// True after initialization is complete
        bool init_complete;
        cv_bridge::CvImageConstPtr cv_rgb;
        cv_bridge::CvImageConstPtr cv_depth;

        // ROS side variables

        /// ROS node for publishers and subscribers
        ros::NodeHandlePtr node_;

        /// Image Transport for RGB image
        std::unique_ptr<image_transport::ImageTransport> rgb_it_;
        /// Image transport for depth image
        std::unique_ptr<image_transport::ImageTransport> depth_it_;

        /// Subscriber with filter for the RGB image
        image_transport::SubscriberFilter rgb_sub;
        /// Subscriber with filter for the depth image
        image_transport::SubscriberFilter depth_sub;
        /// Subscriber with filter for the color camera intrisincs
        message_filters::Subscriber<sensor_msgs::CameraInfo> rgb_info_sub;
        /// Subscriber with filter for the color camera intrisincs
        message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub;

        /// Synchronizer policy typedef for readability
        typedef message_filters::sync_policies::ApproximateTime
            <sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,sensor_msgs::CameraInfo> SyncPolicy;
        /// Messages filters synchronizer
        std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

        /// Publisher for World Rviz markers
        ros::Publisher world_markers_pub;
        /// Publisher for Camera Rviz markers
        ros::Publisher camera_markers_pub;

        /// Rviz World Markers message
        visualization_msgs::Marker world_markers_msg;
        /// Rviz Camera Markers message
        visualization_msgs::Marker camera_markers_msg;

        /// Jaco to camera_link TF
        tf::TransformBroadcaster tf_broadcaster;
        geometry_msgs::Transform tf_camera_transform;
        geometry_msgs::TransformStamped tf_camera_link;


        // ARUCO & Markers variables


        /// Aruco Dictionary
        cv::Ptr<cv::aruco::Dictionary> dictionary;
        /// Detected markers ids in last synchronized callback
        std::vector<int> marker_ids;
        /// Detected marker corners in image plane in latest synchronized callback
        std::vector<std::vector<cv::Point2f>> detected_markers_corners;


        // Horn Transform Variables!


        /// Indicates if the system didnt get enough markers detected in the last iteration
        bool is_lost;
        /// World 3D location of the ARUCO markers
        // const std::map<int,cv::Point3f> all_markers_world = {
        //     {46,cv::Point3f( 0.031, 0.031,0)},    //  XY Plane - Inner
        //     {28,cv::Point3f( 0.209, 0.031,0)},    //  XY Plane - Left
        //     {29,cv::Point3f( 0.031, 0.214,0)},    //  XY Plane - Right
        //     {21,cv::Point3f( 0.217, 0.213,0)},    //  XY Plane - Outer
        //     {39,cv::Point3f( 0.145, 0.143,0)},    //  XY Plane - Center
        //
        //     {42,cv::Point3f(0, 0.031, 0.031)},    //  YZ Plane - Inner
        //     {16,cv::Point3f(0, 0.217, 0.031)},    //  YZ Plane - Left
        //     {24,cv::Point3f(0, 0.031, 0.217)},    //  YZ Plane - Right
        //     {25,cv::Point3f(0, 0.216, 0.216)},    //  YZ Plane - Outer
        //     {47,cv::Point3f(0, 0.145, 0.145)},    //  YZ Plane - Center
        //
        //     {38,cv::Point3f( 0.031,0, 0.031)},    //  ZX Plane - Inner
        //     {35,cv::Point3f( 0.031,0, 0.216)},    //  ZX Plane - Left
        //     {20,cv::Point3f( 0.219,0, 0.031)},    //  ZX Plane - Right
        //     {17,cv::Point3f( 0.217,0, 0.214)},    //  ZX Plane - Outer
        //     {34,cv::Point3f( 0.146,0, 0.144)},    //  ZX Plane - Center
        // };
        const std::map<int,cv::Point3f> all_markers_world = {
            { 0,cv::Point3f( -0.70,     0,0)},
            { 1,cv::Point3f( -0.70, -0.30,0)},
            { 2,cv::Point3f( -0.70, -0.60,0)},
            { 3,cv::Point3f( -0.35,     0,0)},
            { 4,cv::Point3f( -0.35, -0.30,0)},
            { 5,cv::Point3f( -0.35, -0.60,0)},
            { 6,cv::Point3f(   0.0, -0.30,0)},
            { 7,cv::Point3f(   0.0, -0.60,0)},
            { 8,cv::Point3f(  0.35,     0,0)},
            { 9,cv::Point3f(  0.35, -0.30,0)},
            {10,cv::Point3f(  0.35, -0.60,0)},
            {11,cv::Point3f(  0.70,     0,0)},
            {12,cv::Point3f(  0.70, -0.30,0)},
            {13,cv::Point3f(  0.70, -0.60,0)},

            {14,cv::Point3f( 0.043,     0,0.15)},
            {15,cv::Point3f(     0,-0.043,0.15)},
            {16,cv::Point3f(-0.043,     0,0.15)}
        };

        /// Detected markers location in CAMERA frame in meters in latest synchronized callback
        std::vector<cv::Point2f> camera_points;
        /// Detected markers location in WORLD frame in meters
        std::vector<cv::Point3f> world_points;

        cv::Mat camera_intrinsics;
        cv::Mat distortion_coeffs;

        /// Calculated translation
        cv::Mat translation;
        /// Calculated rotation vectors
        cv::Mat rotation_vec;
        /// Rotated into matrix with Rodrigues formuala
        cv::Mat rotation_matrix;

        PnPKstPose KstPose;

    public:
        SingleMarker() : is_lost(true), KstPose("/home/johnmper/.ROSData/single_marker_pose.csv"){
            init_complete = false;

            // Reserving necessary space according to the number of markers defined
            marker_ids.reserve(all_markers_world.size());
            camera_points.reserve(all_markers_world.size());
            world_points.reserve(all_markers_world.size());
            camera_intrinsics = cv::Mat::zeros(3,3,CV_64F);
            distortion_coeffs = cv::Mat::zeros(5,1,CV_64F);

            // Initiate the markers messages for Rviz
            world_markers_msg.header.seq = 0;
            world_markers_msg.header.frame_id = "jaco";
            world_markers_msg.ns = "world_markers";
            world_markers_msg.id = 0;
            world_markers_msg.type = visualization_msgs::Marker::SPHERE_LIST;
            world_markers_msg.action = visualization_msgs::Marker::ADD;
            world_markers_msg.pose.orientation.x = 0;
            world_markers_msg.pose.orientation.y = 0;
            world_markers_msg.pose.orientation.z = 0;
            world_markers_msg.pose.orientation.w = 1;
            world_markers_msg.scale.x = 0.02;
            world_markers_msg.scale.y = 0.02;
            world_markers_msg.scale.z = 0.02;
            world_markers_msg.color.r = 0;
            world_markers_msg.color.g = 0;
            world_markers_msg.color.b = 1;
            world_markers_msg.color.a = 1;
            world_markers_msg.points.resize(all_markers_world.size());
            for( const auto& it : all_markers_world ){
                // Initialize all markers for rviz!
                geometry_msgs::Point point;
                point.x = it.second.x;
                point.y = it.second.y;
                point.z = it.second.z;
                world_markers_msg.points.push_back(point);
            }

            // Initiate the markers messages for Rviz
            camera_markers_msg.header.seq = 0;
            camera_markers_msg.header.frame_id = "rgb_optical_frame";
            camera_markers_msg.ns = "camera_markers";
            camera_markers_msg.id = 0;
            camera_markers_msg.type = visualization_msgs::Marker::SPHERE_LIST;
            camera_markers_msg.action = visualization_msgs::Marker::ADD;
            camera_markers_msg.pose.orientation.x = 0;
            camera_markers_msg.pose.orientation.y = 0;
            camera_markers_msg.pose.orientation.z = 0;
            camera_markers_msg.pose.orientation.w = 1;
            camera_markers_msg.scale.x = 0.05;
            camera_markers_msg.scale.y = 0.05;
            camera_markers_msg.scale.z = 0.05;
            camera_markers_msg.color.r = 0;
            camera_markers_msg.color.g = 1;
            camera_markers_msg.color.b = 0;
            camera_markers_msg.color.a = 1;
            camera_markers_msg.points.reserve(all_markers_world.size());

            // Initialization of Camera TF
            tf_camera_link.header.seq = 0;
            tf_camera_link.header.frame_id = "jaco";
            tf_camera_link.child_frame_id = "r200_camera_link";

            // Send new Camera Link TF
            tf_camera_transform.translation.x = 0;
            tf_camera_transform.translation.y = 0;
            tf_camera_transform.translation.z = 0;
            tf_camera_transform.rotation.x = 0;
            tf_camera_transform.rotation.y = 0;
            tf_camera_transform.rotation.z = 0;
            tf_camera_transform.rotation.w = 1;

            #ifdef SHOW_WINDOW
            // Create image to visualize the marekrs detection
            cv::namedWindow("rgb_image",cv::WINDOW_NORMAL);
            cv::resizeWindow("rgb_image",960,540);
            #endif

            // Initiate ARUCO dictionary
            dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

            start_time_ = ros::Time::now().toSec();
            std::cout << LOG_ID("SingleMarker") << "Created";
        }

        virtual ~SingleMarker(){
            #ifdef SHOW_WINDOW
            cv::destroyAllWindows();
            #endif
        }

    private:
        virtual void onInit(){

            // Temporary node handle
            node_.reset(new ros::NodeHandle());
            ros::NodeHandle& private_nh = getPrivateNodeHandle();

            // Initiate the Publishers for the Rviz markers
            world_markers_pub = node_->advertise<visualization_msgs::Marker>("/p2p/world_markers", 0);
            camera_markers_pub = node_->advertise<visualization_msgs::Marker>("/p2p/camera_markers", 0);

            // Initialize image transports
            rgb_it_.reset(new image_transport::ImageTransport(*node_));
            depth_it_.reset(new image_transport::ImageTransport(*node_));

            // Synchronization initialization, register the synchronized callback (queue_size=1)
            sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(5), rgb_sub, depth_sub, rgb_info_sub, depth_info_sub) );
            sync_->registerCallback(std::bind(&pupil_utils::SingleMarker::synchronizedCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

            // Subscribe to the RGB image topic
            image_transport::TransportHints rgb_hints("raw", ros::TransportHints(), private_nh);\
            rgb_sub.subscribe(*rgb_it_, "/pupil/world/image_raw", 1, rgb_hints);
            if(!rgb_sub.getSubscriber()){
                std::cerr << LOG_ID("SingleMarker") << "Failed to create [/pupil/world/image_raw] subscriber.";
            }
            std::cout << LOG_ID("SingleMarker") << "Subscribed to: " << rgb_sub.getTopic() << std::endl;

            // Subscribe to the depth image topic  with depth image transport (compressedDepth)
            image_transport::TransportHints depth_hints("raw", ros::TransportHints(), private_nh, "depth_image_transport");
            depth_sub.subscribe(*depth_it_, "/pupil/depth_registered/image_raw", 1, depth_hints);
            if(!depth_sub.getSubscriber()){
                std::cerr << LOG_ID("SingleMarker") << "Failed to create [/pupil/depth/image_raw] subscriber.";
            }
            std::cout << LOG_ID("SingleMarker") << "Subscribed to: " << depth_sub.getTopic() << std::endl;

            // Subscribe to the RGB camera_info topic
            rgb_info_sub.subscribe(*node_, "/pupil/world/camera_info",1);
            if(!rgb_info_sub.getSubscriber()){
                std::cerr << LOG_ID("SingleMarker") << "Failed to create [/pupil/world/camera_info] subscriber.";
            }
            std::cout << LOG_ID("SingleMarker") << "Subscribed to: " << rgb_info_sub.getTopic() << std::endl;

            // Subscribe to the Depth camera info topic
            depth_info_sub.subscribe(*node_, "/pupil/depth/camera_info",1);
            if(!depth_info_sub.getSubscriber()){
                std::cerr << LOG_ID("SingleMarker") << "Failed to create [/pupil/depth/camera_info] subscriber.";
            }
            std::cout << LOG_ID("SingleMarker") << "Subscribed to: " << depth_info_sub.getTopic() << std::endl;

            init_complete = true;
            std::cout << LOG_ID("SingleMarker") << "Initialization Complete" << std::endl;

        }

        /// synchronized callback!!! Everything that is goog comes from this function!! MUAHAHA
        void synchronizedCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                                  const sensor_msgs::ImageConstPtr& depth_msg,
                                  const sensor_msgs::CameraInfoConstPtr& rgb_info_msg,
                                  const sensor_msgs::CameraInfoConstPtr& depth_info_msg){

            if(!init_complete){
                std::cout << LOG_ID("SingleMarker") << " Synchronized callback called before initilization was completed.";
                return;
            }

            #ifdef TIME_SINGLE
            unsigned int time_detect_markers = 0;
            unsigned int time_pose_estimation = 0;
            unsigned int time_callback = 0;
            std::chrono::high_resolution_clock::time_point cb_t1, cb_t2,local_t1, local_t2;
            cb_t1 = hr_clock::now();
            #endif

            // Turn the color image into a CV Matriz, necessary for the
            try {
                cv_rgb = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
            }
            catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            // Update camera parameters
            camera_intrinsics.at<double>(0,0) = (float)rgb_info_msg->K[0];
            camera_intrinsics.at<double>(0,1) = (float)rgb_info_msg->K[1];
            camera_intrinsics.at<double>(0,2) = (float)rgb_info_msg->K[2];
            camera_intrinsics.at<double>(1,0) = (float)rgb_info_msg->K[3];
            camera_intrinsics.at<double>(1,1) = (float)rgb_info_msg->K[4];
            camera_intrinsics.at<double>(1,2) = (float)rgb_info_msg->K[5];
            camera_intrinsics.at<double>(2,0) = (float)rgb_info_msg->K[6];
            camera_intrinsics.at<double>(2,1) = (float)rgb_info_msg->K[7];
            camera_intrinsics.at<double>(2,2) = (float)rgb_info_msg->K[8];

            distortion_coeffs.at<double>(0,0) = (float)rgb_info_msg->D[0];
            distortion_coeffs.at<double>(1,0) = (float)rgb_info_msg->D[1];
            distortion_coeffs.at<double>(2,0) = (float)rgb_info_msg->D[2];
            distortion_coeffs.at<double>(3,0) = (float)rgb_info_msg->D[3];
            distortion_coeffs.at<double>(4,0) = (float)rgb_info_msg->D[4];

            #ifdef TIME_SINGLE
            local_t1 = hr_clock::now();
            #endif
            // Detect the position in the color image of the aruco markers
            cv::aruco::detectMarkers(cv_rgb->image, dictionary, detected_markers_corners, marker_ids);
            #ifdef TIME_SINGLE
            local_t2 = hr_clock::now();
            time_detect_markers = std::chrono::duration_cast<std::chrono::microseconds>( local_t2 - local_t1 ).count();
            #endif

            #ifdef SHOW_WINDOW
            cv::Mat markers_image;
            #endif

            #ifdef TIME_SINGLE
            local_t1 = hr_clock::now();
            #endif
            if(marker_ids.size() > 0){
                std::vector<cv::Vec3d> rvecs, tvecs;
                cv::aruco::estimatePoseSingleMarkers(detected_markers_corners, 0.056, camera_intrinsics, distortion_coeffs, rvecs, tvecs);

                rotation_vec = cv::Mat(0,0,0);
                translation = cv::Mat(0,0,0);
                for(unsigned int ii=0; ii < marker_ids.size(); ii++){

                    rotation_vec = rotation_vec + rvecs[ii];
                    translation = translation + tvecs[ii];
                }
                rotation_vec /= (float)marker_ids.size();
                translation /= (float)marker_ids.size();

                #ifdef SHOW_WINDOW
                cv_rgb->image.copyTo(markers_image);
                cv::aruco::drawDetectedMarkers(markers_image, detected_markers_corners, marker_ids);
                for(unsigned int ii=0; ii < marker_ids.size(); ii++){
                    std::cout << "  R:\n" << rvecs[ii] << "\n  T:\n" << tvecs[ii] << "\n-----------" <<std::endl;
                    cv::aruco::drawAxis(markers_image, camera_intrinsics, distortion_coeffs, rvecs[ii], tvecs[ii], 0.1);
                }
                #endif
            }

            // // Clean detected markers from last iteration
            // ++tf_camera_link.header.seq;
            //
            // cv::Rodrigues(rotation_vec, rotation_matrix);
            // cv::Mat zrot;
            // cv::Mat yrot;
            // cv::Mat tr;
            // yrot = (cv::Mat_<double>(3,3) << cos(-math::pi/2.0), 0,-sin(-math::pi/2.0),0,1,0,sin(-math::pi/2.0),0,cos(-math::pi/2.0) );
            // zrot = (cv::Mat_<double>(3,3) << cos(math::pi/2.0), sin(math::pi/2.0),0,-sin(math::pi/2.0),cos(math::pi/2.0),0,0,0,1 );
            //
            // tr = -rotation_matrix.t()*translation;
            // rotation_matrix = yrot*zrot*rotation_matrix;
            //
            // tf_camera_transform.rotation.w = sqrt(1 + rotation_matrix.at<double>(0,0) + rotation_matrix.at<double>(1,1) + rotation_matrix.at<double>(2,2))/2.0;
            // tf_camera_transform.rotation.x = (rotation_matrix.at<double>(1,2)-rotation_matrix.at<double>(2,1))/(4*tf_camera_transform.rotation.w);
            // tf_camera_transform.rotation.y = (rotation_matrix.at<double>(2,0)-rotation_matrix.at<double>(0,2))/(4*tf_camera_transform.rotation.w);
            // tf_camera_transform.rotation.z = (rotation_matrix.at<double>(0,1)-rotation_matrix.at<double>(1,0))/(4*tf_camera_transform.rotation.w);
            //
            // if(tf_camera_transform.rotation.w < 0){
            //     tf_camera_transform.rotation.w = -tf_camera_transform.rotation.w;
            //     tf_camera_transform.rotation.x = -tf_camera_transform.rotation.x;
            //     tf_camera_transform.rotation.y = -tf_camera_transform.rotation.y;
            //     tf_camera_transform.rotation.z = -tf_camera_transform.rotation.z;
            // }
            // tf_camera_transform.translation.x = tr.at<double>(0);
            // tf_camera_transform.translation.y = tr.at<double>(1);
            // tf_camera_transform.translation.z = tr.at<double>(2);
            //
            #ifdef TIME_SINGLE
            local_t2 = hr_clock::now();
            time_pose_estimation = std::chrono::duration_cast<std::chrono::microseconds>( local_t2 - local_t1 ).count();
            std::cout << "SingleMarker Duration: " << time_pose_estimation << std::endl;
            #endif

            // std::cout << tf_camera_link << std::endl;
            //
            // tf_camera_link.header.stamp = ros::Time::now();
            // tf_camera_link.transform = tf_camera_transform;
            // tf_broadcaster.sendTransform(tf_camera_link);
            //
            //
            #ifdef TIME_SINGLE
            cb_t2 = hr_clock::now();
            auto cb_duration = std::chrono::duration_cast<std::chrono::microseconds>( cb_t2 - cb_t1 ).count();
            if(tf_camera_link.header.stamp.toSec()-start_time_ < 50){
                KstPose.add(tf_camera_link.header.stamp.toSec()-start_time_, time_detect_markers, time_pose_estimation,tf_camera_transform);
            }
            else{
                std::cout << "enough" << std::endl;
            }
            #endif

            #ifdef SHOW_WINDOW
            // Show color image
            cv::imshow("rgb_image",markers_image);
            cv::waitKey(1);
            #endif
        }


    };

    PLUGINLIB_DECLARE_CLASS(pupil_utils, SingleMarker, pupil_utils::SingleMarker, nodelet::Nodelet);
}
