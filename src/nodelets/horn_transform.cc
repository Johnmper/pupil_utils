/**
@author     johnmper
@file       horn_transform.cc
@brief      ROS nodelet that calculates the transform between the JACO robot and R200 camera using Horn algorithm.
@details    ROS nodelet that calculates the transform between the JACO robot and R200 camera. Subscribes to
    [pupil/world/image_raw] and [pupil/depth/image_raw] uses fiducial markers in the JACO robot to calculate
    using the horn algorithm the ideal Pose for the camera.
*/

// C++ STANDARD INCLUDES
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <utility>
#include <map>
#include <tuple>
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
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Eigen>

#include <ros_utils/style.hh>
#include <ros_utils/filter.hh>

using hr_clock = std::chrono::high_resolution_clock;
#define MS_CAST(a,b) std::chrono::duration_cast<std::chrono::microseconds>(a-b)

#undef TIME_HORN
#undef DEBUG_HORN

// CHANGED: ADDED TIME_FUNCTION_DEBUG()!!
#ifdef TIME_HORN
std::chrono::high_resolution_clock::time_point t1,t2;
double debug_duration=0.0;
#define TIME_FUNCTION_DEBUG(a) \
    t1 = std::chrono::high_resolution_clock::now();\
    a;\
    t2 = std::chrono::high_resolution_clock::now();\
    debug_duration = (double)std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();\
    std::cout << "Took: " << debug_duration << " us" << std::endl;
#endif


namespace pupil_utils{

    struct HornKstPose{
        std::string filepath;
        std::ofstream file;

        HornKstPose(std::string filepath_) : filepath(filepath_){
            file.open(filepath,std::ofstream::out);
            if(file.fail()){
                std::cerr << "Failed to create " << filepath << " file." << std::endl;
            }else{
                file << "time, detect_duration, process_duration, tx, ty, tz, qx, qy, qz, qw";
                file << std::endl;
            }
        }
        ~HornKstPose(){
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
//   #     #                         #######
//   #     #  ####  #####  #    #       #    #####    ##   #    #  ####  ######  ####  #####  #    #
//   #     # #    # #    # ##   #       #    #    #  #  #  ##   # #      #      #    # #    # ##  ##
//   ####### #    # #    # # #  #       #    #    # #    # # #  #  ####  #####  #    # #    # # ## #
//   #     # #    # #####  #  # #       #    #####  ###### #  # #      # #      #    # #####  #    #
//   #     # #    # #   #  #   ##       #    #   #  #    # #   ## #    # #      #    # #   #  #    #
//   #     #  ####  #    # #    #       #    #    # #    # #    #  ####  #       ####  #    # #    #
//===========================================================================================================
    class HornTransform: public nodelet::Nodelet{
    private:




        /// True after initialization is complete
        bool init_complete;
        /// Path to kst save dir
        const std::string save_dir = "/home/johnmper/.ROSData/pupil/";
        // ROS side variables

        /// ROS node for publishers and subscribers
        ros::NodeHandlePtr node_;
        /// ROS time when nodelet was first initialized
        double start_time;
        /// Current Time since nodelet initiated
        double current_time;

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

        /// Pose Publisher in case message is prefered instead of TF
        ros::Publisher horn_pose_pub;

        /// Publisher for World Rviz markers
        ros::Publisher world_markers_pub;
        /// Publisher for Camera Rviz markers
        ros::Publisher camera_markers_pub;

        /// Rviz World Markers message
        visualization_msgs::Marker world_markers_msg;
        /// Rviz Camera Markers message
        visualization_msgs::Marker camera_markers_msg;

        /// [jaco -> camera_link] transform publisher
        tf::TransformBroadcaster tf_broadcaster;
        /// Smaller piece of the Transform message
        geometry_msgs::Transform tf_camera_transform;
        /// Actual [jaco -> camera_link] transform message
        geometry_msgs::TransformStamped tf_camera_link;


        // ARUCO & Markers variables


        cv::Ptr<cv::aruco::DetectorParameters> parameters;
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
        const std::map<int,Eigen::Matrix<float,3,1>> all_markers_world = {
            { 0,Eigen::Matrix<float,3,1>({ -0.70,     0,0})},
            { 1,Eigen::Matrix<float,3,1>({ -0.70, -0.30,0})},
            { 2,Eigen::Matrix<float,3,1>({ -0.70, -0.60,0})},
            { 3,Eigen::Matrix<float,3,1>({ -0.35,     0,0})},
            { 4,Eigen::Matrix<float,3,1>({ -0.35, -0.30,0})},
            { 5,Eigen::Matrix<float,3,1>({ -0.35, -0.60,0})},
            { 6,Eigen::Matrix<float,3,1>({   0.0, -0.30,0})},
            { 7,Eigen::Matrix<float,3,1>({   0.0, -0.60,0})},
            { 8,Eigen::Matrix<float,3,1>({  0.35,     0,0})},
            { 9,Eigen::Matrix<float,3,1>({  0.35, -0.30,0})},
            {10,Eigen::Matrix<float,3,1>({  0.35, -0.60,0})},
            {11,Eigen::Matrix<float,3,1>({  0.70,     0,0})},
            {12,Eigen::Matrix<float,3,1>({  0.70, -0.30,0})},
            {13,Eigen::Matrix<float,3,1>({  0.70, -0.60,0})},

            {14,Eigen::Matrix<float,3,1>({ 0.043,     0,0.15})},
            {15,Eigen::Matrix<float,3,1>({     0,-0.043,0.15})},
            {16,Eigen::Matrix<float,3,1>({-0.043,     0,0.15})},

            // BOX Markers! SIZE:
            // X = 0.225, Y = 0.197, Z = 0.069
            // Distance from corners: 0.032
            {18,Eigen::Matrix<float,3,1>({-0.0805, -0.0665, +0.0345})},
            {19,Eigen::Matrix<float,3,1>({-0.0805, +0.0665, +0.0345})},
            {20,Eigen::Matrix<float,3,1>({+0.0805, -0.0665, +0.0345})},
            {21,Eigen::Matrix<float,3,1>({+0.0805, +0.0665, +0.0345})},
            {22,Eigen::Matrix<float,3,1>({-0.0805, +0.0985, +0.0000})},
            {23,Eigen::Matrix<float,3,1>({+0.0805, +0.0985, +0.0000})},
            {24,Eigen::Matrix<float,3,1>({-0.1125, -0.0665, +0.0000})},
            {25,Eigen::Matrix<float,3,1>({-0.1125, +0.0665, +0.0000})},
            {26,Eigen::Matrix<float,3,1>({+0.1125, +0.0665, +0.0000})},
            {27,Eigen::Matrix<float,3,1>({+0.1125, -0.0665, +0.0000})},
            {28,Eigen::Matrix<float,3,1>({-0.0805, -0.0665, -0.0345})},
            {29,Eigen::Matrix<float,3,1>({-0.0805, +0.0665, -0.0345})},
            {30,Eigen::Matrix<float,3,1>({+0.0805, -0.0665, -0.0345})},
            {31,Eigen::Matrix<float,3,1>({+0.0805, +0.0665, -0.0345})}
        };


        // const std::map<int,Eigen::Matrix<float,3,1>> all_markers_world = {
        //     { 0,Eigen::Matrix<float,3,1>({ 0.031, 0.031,0})},    //  XY Plane - Inner
        //     { 1,Eigen::Matrix<float,3,1>({ 0.209, 0.031,0})},    //  XY Plane - Left
        //     { 2,Eigen::Matrix<float,3,1>({ 0.031, 0.214,0})},    //  XY Plane - Right
        //     { 3,Eigen::Matrix<float,3,1>({ 0.217, 0.213,0})},    //  XY Plane - Outer
        //     { 4,Eigen::Matrix<float,3,1>({ 0.145, 0.143,0})},    //  XY Plane - Center
        //
        //     { 5,Eigen::Matrix<float,3,1>({0, 0.031, 0.031})},    //  YZ Plane - Inner
        //     { 6,Eigen::Matrix<float,3,1>({0, 0.217, 0.031})},    //  YZ Plane - Left
        //     { 7,Eigen::Matrix<float,3,1>({0, 0.031, 0.217})},    //  YZ Plane - Right
        //     { 8,Eigen::Matrix<float,3,1>({0, 0.216, 0.216})},    //  YZ Plane - Outer
        //     { 9,Eigen::Matrix<float,3,1>({0, 0.145, 0.145})},    //  YZ Plane - Center
        //
        //     {10,Eigen::Matrix<float,3,1>({ 0.031, 0, 0.031})},    //  ZX Plane - Inner
        //     {11,Eigen::Matrix<float,3,1>({ 0.031, 0, 0.216})},    //  ZX Plane - Left
        //     {12,Eigen::Matrix<float,3,1>({ 0.219, 0, 0.031})},    //  ZX Plane - Right
        //     {13,Eigen::Matrix<float,3,1>({ 0.217, 0, 0.214})},    //  ZX Plane - Outer
        //     {14,Eigen::Matrix<float,3,1>({ 0.146, 0, 0.144})},    //  ZX Plane - Center
        //
        //     {18,Eigen::Matrix<float,3,1>({-0.0805, -0.0665, +0.0345})},
        //     {19,Eigen::Matrix<float,3,1>({-0.0805, +0.0665, +0.0345})},
        //     {20,Eigen::Matrix<float,3,1>({+0.0805, -0.0665, +0.0345})},
        //     {21,Eigen::Matrix<float,3,1>({+0.0805, +0.0665, +0.0345})},
        //     {22,Eigen::Matrix<float,3,1>({-0.0805, +0.0985, +0.0000})},
        //     {23,Eigen::Matrix<float,3,1>({+0.0805, +0.0985, +0.0000})},
        //     {24,Eigen::Matrix<float,3,1>({-0.1125, -0.0665, +0.0000})},
        //     {25,Eigen::Matrix<float,3,1>({-0.1125, +0.0665, +0.0000})},
        //     {26,Eigen::Matrix<float,3,1>({+0.1125, +0.0665, +0.0000})},
        //     {27,Eigen::Matrix<float,3,1>({+0.1125, -0.0665, +0.0000})},
        //     {28,Eigen::Matrix<float,3,1>({-0.0805, -0.0665, -0.0345})},
        //     {29,Eigen::Matrix<float,3,1>({-0.0805, +0.0665, -0.0345})},
        //     {30,Eigen::Matrix<float,3,1>({+0.0805, -0.0665, -0.0345})},
        //     {31,Eigen::Matrix<float,3,1>({+0.0805, +0.0665, -0.0345})}
        // };

        /// Estimate all World markers location based last iteration calculated pose
        std::map<int, float> estimated_markers_depth;
        /// Detected points centroid in CAMERA frame
        Eigen::Matrix<float,3,1> camera_centroid;
        /// Detected markers location in CAMERA frame in meters in latest synchronized callback
        std::vector<Eigen::Matrix<float,3,1>> camera_points;
        /// Detected markers vectors from centroid to points in CAMERA frame
        std::vector<Eigen::Matrix<float,3,1>> camera_vectors;

        /// Detected points centroid in WORLD frame
        Eigen::Matrix<float,3,1> world_centroid;
        /// Detected markers location in WORLD frame in meters
        std::vector<Eigen::Matrix<float,3,1>> world_points;
        /// Detected markers vectors from centroid to points in WORLD frame
        std::vector<Eigen::Matrix<float,3,1>> world_vectors;

        Eigen::EigenSolver<Eigen::Matrix<float,4,4>> eig_solver;
        /// Scale factor for the Horn algorithm
        float scale;
        /// Calculated translation
        Eigen::Matrix<float,3,1> translation;
        /// Calculated rotationdepth_info_sub
        Eigen::Quaternion<float> rotation;

        // NOTE:
        std::vector<int> accepted_markers;
        Eigen::Matrix<float,3,1> proj_error;


        /// BOX Variables

        ros::Publisher              box_horn_pub;
        // Markers for visualization in rviz!
        ros::Publisher              camera_box_markers_pub;
        visualization_msgs::Marker  camera_box_markers_msg;

        // Markers detected in world variables
        std::vector<int>            accepted_box_markers;
        Eigen::Matrix<float,3,1>    camera_box_centroid;
        Eigen::Matrix<float,3,1>    world_box_centroid;
        std::vector<Eigen::Matrix<float,3,1>> camera_box_points;
        std::vector<Eigen::Matrix<float,3,1>> camera_box_vectors;
        std::vector<Eigen::Matrix<float,3,1>> world_box_points;
        std::vector<Eigen::Matrix<float,3,1>> world_box_vectors;

        // Final Calculated Pose for Box!
        Eigen::Matrix<float,3,1>    box_translation;
        Eigen::Quaternion<float>    box_rotation;

        // Pose TF variables
        geometry_msgs::Transform        tf_box_transform;
        geometry_msgs::TransformStamped tf_box_link;

        bool box_is_lost;

        HornKstPose KstPose;
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
        HornTransform() : is_lost(true), box_is_lost(true), KstPose("/home/johnmper/.ROSData/horn_pose.csv"){
            init_complete = false;
            // Reserving necessary space according to the number of markers defined
            marker_ids.reserve(all_markers_world.size());
            camera_points.reserve(all_markers_world.size());
            camera_vectors.reserve(all_markers_world.size());
            world_points.reserve(all_markers_world.size());
            world_vectors.reserve(all_markers_world.size());

            for(const auto& it : all_markers_world){
                estimated_markers_depth.insert({it.first,0});
            }
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
            world_markers_msg.scale.x = 0.04;
            world_markers_msg.scale.y = 0.04;
            world_markers_msg.scale.z = 0.04;
            world_markers_msg.color.r = 0;
            world_markers_msg.color.g = 0;
            world_markers_msg.color.b = 1;
            world_markers_msg.color.a = 0.6;
            world_markers_msg.points.reserve(all_markers_world.size());
            for( const auto& it : all_markers_world ){
                // Initialize all markers for rviz!
                if(it.first < 18){

                    geometry_msgs::Point point;
                    point.x = it.second.coeffRef(0);
                    point.y = it.second.coeffRef(1);
                    point.z = it.second.coeffRef(2);
                    world_markers_msg.points.push_back(point);
                }
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
            camera_markers_msg.scale.x = 0.04;
            camera_markers_msg.scale.y = 0.04;
            camera_markers_msg.scale.z = 0.04;
            camera_markers_msg.color.r = 0;
            camera_markers_msg.color.g = 1;
            camera_markers_msg.color.b = 0;
            camera_markers_msg.color.a = 0.6;
            camera_markers_msg.points.reserve(all_markers_world.size());

            // Initialization of Camera TF
            tf_camera_link.header.frame_id = "jaco";
            tf_camera_link.child_frame_id = "r200_camera_link";

            // Initialization of the camera link translation and rotation
            translation << 0,0,0;
            rotation.w() = 1;
            rotation.x() = 0;
            rotation.y() = 0;
            rotation.z() = 0;


            // Initialization for BOX Markers variables
            camera_box_points.reserve(all_markers_world.size());
            camera_box_vectors.reserve(all_markers_world.size());
            world_box_points.reserve(all_markers_world.size());
            world_box_vectors.reserve(all_markers_world.size());

            camera_box_markers_msg.header.seq = 0;
            camera_box_markers_msg.header.frame_id = "rgb_optical_frame";
            camera_box_markers_msg.ns = "box_markers";
            camera_box_markers_msg.id = 0;
            camera_box_markers_msg.type = visualization_msgs::Marker::SPHERE_LIST;
            camera_box_markers_msg.action = visualization_msgs::Marker::ADD;
            camera_box_markers_msg.pose.orientation.x = 0;
            camera_box_markers_msg.pose.orientation.y = 0;
            camera_box_markers_msg.pose.orientation.z = 0;
            camera_box_markers_msg.pose.orientation.w = 1;
            camera_box_markers_msg.scale.x = 0.04;
            camera_box_markers_msg.scale.y = 0.04;
            camera_box_markers_msg.scale.z = 0.04;
            camera_box_markers_msg.color.r = 1;
            camera_box_markers_msg.color.g = 0;
            camera_box_markers_msg.color.b = 1;
            camera_box_markers_msg.color.a = 1;
            camera_box_markers_msg.points.reserve(16);
            tf_box_link.header.frame_id = "r200_camera_link";
            tf_box_link.child_frame_id = "box_link";
            // Initialization of the camera link translation and rotation
            box_translation << 0,0,0;
            box_rotation.w() = 1;
            box_rotation.x() = 0;
            box_rotation.y() = 0;
            box_rotation.z() = 0;

            #ifdef DEBUG_HORN
            // Create image to visualize the marekrs detection
            cv::namedWindow("rgb_image",cv::WINDOW_NORMAL);
            cv::resizeWindow("rgb_image",960,540);
            #endif

            // Initiate ARUCO dictionary
            dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

            parameters = cv::aruco::DetectorParameters::create();
            std::cout << parameters->maxMarkerPerimeterRate << std::endl;
            std::cout << parameters->minMarkerPerimeterRate << std::endl;
            // Higher value, for 5cm markers means that it wont be caught by the depth frame
            parameters->maxMarkerPerimeterRate = 0.2;

            parameters->adaptiveThreshWinSizeMin = 13;
            parameters->adaptiveThreshWinSizeMax = 33;
            parameters->adaptiveThreshWinSizeStep = 20;

            std::cout << LOG_ID("HornTransform") << "Created" << std::endl;

        }

        virtual ~HornTransform(){
            cv::destroyAllWindows();
        }


//===========================================================================================================
//                 ###
//    ####  #    #  #  #    # # #####
//   #    # ##   #  #  ##   # #   #
//   #    # # #  #  #  # #  # #   #
//   #    # #  # #  #  #  # # #   #
//   #    # #   ##  #  #   ## #   #
//    ####  #    # ### #    # #   #
//===========================================================================================================
    private:
        virtual void onInit(){

            // Temporary node handle
            node_.reset(new ros::NodeHandle());
            ros::NodeHandle& private_nh = getPrivateNodeHandle();

            // Initiate the Publishers for the Rviz markers
            world_markers_pub = node_->advertise<visualization_msgs::Marker>("/horn/world_markers", 0);
            camera_markers_pub = node_->advertise<visualization_msgs::Marker>("/horn/camera_markers", 0);
            horn_pose_pub = node_->advertise<geometry_msgs::PoseStamped>("/horn/pose", 0);

            // CHANGED:
            box_horn_pub = node_->advertise<geometry_msgs::PoseStamped>("/horn/box",0);
            camera_box_markers_pub = node_->advertise<visualization_msgs::Marker>("/horn/box_markers", 0);


            // Initialize image transports
            rgb_it_.reset(new image_transport::ImageTransport(*node_));
            depth_it_.reset(new image_transport::ImageTransport(*node_));

            // Synchronization initialization, register the synchronized callback (queue_size=1)
            sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(5), rgb_sub, depth_sub, rgb_info_sub, depth_info_sub) );
            sync_->registerCallback(std::bind(&pupil_utils::HornTransform::synchronizedCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

            // Subscribe to the RGB image topic
            image_transport::TransportHints rgb_hints("raw", ros::TransportHints(), private_nh);
            rgb_sub.subscribe(*rgb_it_, "/pupil/world/image_raw", 1, rgb_hints);
            if(!rgb_sub.getSubscriber()){
                std::cerr << LOG_ID("HornTransform") << "Failed to create [/pupil/world/image_raw] subscriber.";
            }
            std::cout << LOG_ID("HornTransform") << "Subscribed to: " << rgb_sub.getTopic() << std::endl;

            // Subscribe to the depth image topic  with depth image transport (compressedDepth)
            image_transport::TransportHints depth_hints("raw", ros::TransportHints(), private_nh, "depth_image_transport");
            depth_sub.subscribe(*depth_it_, "/pupil/depth_registered/image_raw", 1, depth_hints);
            if(!depth_sub.getSubscriber()){
                std::cerr << LOG_ID("HornTransform") << "Failed to create [/pupil/depth/image_raw] subscriber.";
            }
            std::cout << LOG_ID("HornTransform") << "Subscribed to: " << depth_sub.getTopic() << std::endl;

            // Subscribe to the RGB camera_info topic
            rgb_info_sub.subscribe(*node_, "/pupil/world/camera_info",1);
            if(!rgb_info_sub.getSubscriber()){
                std::cerr << LOG_ID("HornTransform") << "Failed to create [/pupil/world/camera_info] subscriber.";
            }
            std::cout << LOG_ID("HornTransform") << "Subscribed to: " << rgb_info_sub.getTopic() << std::endl;

            // Subscribe to the Depth camera info topic
            depth_info_sub.subscribe(*node_, "/pupil/depth/camera_info",1);
            if(!depth_info_sub.getSubscriber()){
                std::cerr << LOG_ID("HornTransform") << "Failed to create [/pupil/depth/camera_info] subscriber.";
            }
            std::cout << LOG_ID("HornTransform") << "Subscribed to: " << depth_info_sub.getTopic() << std::endl;

            // Store Ros time when nodelet initiated
            start_time = ros::Time::now().toSec();

            // Initialize Kst files
            initKst();

            // Initialization complete. Upi!
            init_complete = true;
            std::cout << LOG_ID("HornTransform") << "Initialization Complete" << std::endl;
        }//end onInitpupil/world



//===========================================================================================================
//    #####                                                                          #####
//   #     # #   # #    #  ####  #    # #####   ####  #    # # ###### ###### #####  #     #   ##   #      #      #####    ##    ####  #    #
//   #        # #  ##   # #    # #    # #    # #    # ##   # #     #  #      #    # #        #  #  #      #      #    #  #  #  #    # #   #
//    #####    #   # #  # #      ###### #    # #    # # #  # #    #   #####  #    # #       #    # #      #      #####  #    # #      ####
//         #   #   #  # # #      #    # #####  #    # #  # # #   #    #      #    # #       ###### #      #      #    # ###### #      #  #
//   #     #   #   #   ## #    # #    # #   #  #    # #   ## #  #     #      #    # #     # #    # #      #      #    # #    # #    # #   #
//    #####    #   #    #  ####  #    # #    #  ####  #    # # ###### ###### #####   #####  #    # ###### ###### #####  #    #  ####  #    #
//===========================================================================================================
        void synchronizedCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                                  const sensor_msgs::ImageConstPtr& depth_msg,
                                  const sensor_msgs::CameraInfoConstPtr& rgb_info_msg,
                                  const sensor_msgs::CameraInfoConstPtr& depth_info_msg){
            static cv_bridge::CvImageConstPtr cv_rgb;

            #ifdef TIME_HORN
            unsigned int time_detect_markers = 0;
            unsigned int time_callback = 0;
            unsigned int time_pose_estimation = 0;
            std::chrono::high_resolution_clock::time_point cb_t1, cb_t2, local_t1, local_t2;;
            cb_t1 = hr_clock::now();
            #endif

            #ifdef DEBUG_HORN
            static int kk = 0;
            std::cout << "FRAME NUMBER: " << kk++ << std::endl;
            #endif

            if(!init_complete){
                std::cout << LOG_ID("HornTransform") << " Synchronized callback called before initilization was completed.";
                return;
            }

            current_time = ros::Time::now().toSec()-start_time;


            #ifdef TIME_HORN
            local_t1 = hr_clock::now();
            #endif
            // Turn the color image into a CV Matriz, necessary for the
            try {
                cv_rgb = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
            }
            catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            // Detect the position in the color image of the aruco markers
            cv::aruco::detectMarkers(cv_rgb->image, dictionary, detected_markers_corners, marker_ids, parameters);
            #ifdef TIME_HORN
            local_t2 = hr_clock::now();
            time_detect_markers = std::chrono::duration_cast<std::chrono::microseconds>( local_t2 - local_t1 ).count();
            #endif

            #ifdef DEBUG_HORN
            // NOTE: Remove in final version!
            cv::Mat markers_image;
            cv_rgb->image.copyTo(markers_image);
            cv::aruco::drawDetectedMarkers(markers_image, detected_markers_corners, marker_ids);
            #endif

            // Clean detected markers from last iteration
            ++tf_camera_link.header.seq;
            ++world_markers_msg.header.seq;
            ++camera_markers_msg.header.seq;

            // CHANGED:
            ++tf_box_link.header.seq;
            ++camera_box_markers_msg.header.seq;

            // Only calculate the current camera pose if more than 3 markers where detected!@!!
            if(marker_ids.size() >= 3){
                #ifdef DEBUG_HORN
                std::cout << style::yellow << style::bold << " Detected " << marker_ids.size() << " markers" << std::endl << style::normal;
                #endif


                world_centroid << Eigen::Matrix<float,3,1>::Zero();
                camera_centroid << Eigen::Matrix<float,3,1>::Zero();
                world_points.clear();
                camera_points.clear();
                camera_markers_msg.points.clear();

                // CHANGED:
                world_box_centroid << Eigen::Matrix<float,3,1>::Zero();
                camera_box_centroid << Eigen::Matrix<float,3,1>::Zero();
                world_box_points.clear();
                camera_box_points.clear();
                camera_box_markers_msg.points.clear();

                for(size_t ii=0; ii < marker_ids.size();++ii){
                    // Check if detected marker has been defined
                    if(all_markers_world.find(marker_ids[ii]) == all_markers_world.end()){
                        continue;
                    }

                    // Create temporary 3D point
                    geometry_msgs::Point point;

                    // Calculate the center of the marker from the four corners
                    unsigned int rgb_u = (unsigned int)std::round((detected_markers_corners[ii][0].x + detected_markers_corners[ii][1].x
                                + detected_markers_corners[ii][2].x + detected_markers_corners[ii][3].x)/4.0);
                    unsigned int rgb_v = (unsigned int)std::round((detected_markers_corners[ii][0].y + detected_markers_corners[ii][1].y
                                + detected_markers_corners[ii][2].y + detected_markers_corners[ii][3].y)/4.0);


                    // Get depth mean from the are surrounding the (u,v) coordiante in the depth_registered frame
                    point.z = 0;
                    unsigned int n_valids = 0;
                    unsigned int sz = 15;
                    unsigned int step = depth_msg->width;
                    uint16_t* reg_depth = (uint16_t*)(&depth_msg->data[0]);
                    for(unsigned int ii = rgb_v-sz; ii <= rgb_v+sz; ++ii){
                        for(unsigned int jj = rgb_u-sz; jj <= rgb_u+sz; ++jj){
                            uint16_t& T = reg_depth[ii*step + jj];
                            if(T > 750U && T < 3400U){
                                point.z += T*0.001;
                                ++n_valids;
                            }
                        }
                    }

                    if(n_valids <= 17){
                        #ifdef DEBUG_HORN
                        std::cout << style::red << "     No Depth Found" << style::normal << ": ("<<marker_ids[ii] <<")" <<std::endl;
                        #endif
                        continue;
                    }

                    // // Calculate the 3D depth distance mean
                    point.z /= (float)n_valids;
                    double p0 = 0.09327101;
                    double p1 = 0.83614111*point.z;
                    double p2 = 0.07983241*point.z*point.z;
                    double p3 = -0.01575206*point.z*point.z*point.z;
                    point.z = p0+p1+p2+p3;

                    // Calculate the projections from depth image into the 3D camera points
                    point.x = (rgb_u-rgb_info_msg->K[2])*point.z/rgb_info_msg->K[0];
                    point.y = (rgb_v-rgb_info_msg->K[5])*point.z/rgb_info_msg->K[4];

                    if( 0 <= marker_ids[ii] && marker_ids[ii] <= 16 ){
                        if(!is_lost){
                            if( fabs(point.z-estimated_markers_depth[marker_ids[ii]]) > 0.25 ){
                                #ifdef DEBUG_HORN
                                std::cout << style::yellow << "     Invalid Depth transition"<< style::normal <<": ("<< marker_ids[ii]<<") : " << estimated_markers_depth[marker_ids[ii]] << " -> "<< point.z << std::endl;
                                #endif
                                continue;
                            }
                        }
                    }

                    #ifdef DEBUG_HORN
                    std::cout << "     " << style::green << style::bold << marker_ids[ii] << style::normal << " -> (" << point.x << ", " << point.y << ", " << point.z << ") " <<std::endl<<std::flush;
                    #endif

                    /// Check if it is world marker of Object
                    if( 0 <= marker_ids[ii] && marker_ids[ii] <= 16 ){
                        // Add calculated 3D point into the markers in camera frame
                        camera_markers_msg.points.push_back(point);

                        // Update the camera point (NOTE: Changed Z -> X, -X -> Y, -Y -> Z) to transform the point in optical frame to the R200_Camera_Frame
                        camera_points.push_back(Eigen::Matrix<float,3,1>({(float)point.z,(float)-point.x,(float)-point.y}));

                        // Add detected marker world point into the list
                        world_points.push_back(all_markers_world.at(marker_ids[ii]));

                        accepted_markers.push_back(marker_ids[ii]);
                        // Sum all elements
                        camera_centroid += camera_points.back();
                        world_centroid += world_points.back();
                    }
                    else if( 18 <= marker_ids[ii] && marker_ids[ii] <= 31 ){
                        // Box object, update the object 3D points message
                        camera_box_markers_msg.points.push_back(point);
                        accepted_box_markers.push_back(marker_ids[ii]);

                        camera_box_points.push_back(Eigen::Matrix<float,3,1>({(float)point.z,(float)-point.x,(float)-point.y}));
                        world_box_points.push_back(all_markers_world.at(marker_ids[ii]));

                        camera_box_centroid += camera_box_points.back();
                        world_box_centroid += world_box_points.back();
                    }

                }

                #ifdef DEBUG_HORN
                std::cout << " Valid Markers: " << style::yellow << style::bold << "(" << world_points.size() << "/" << marker_ids.size() << ") "<<style::normal;
                std::cout << " Valid Box Markers: " << style::yellow << style::bold << "(" << world_box_points.size() << "/" << marker_ids.size() << ") "<<style::normal;
                #endif

                // Check if there are enough valid depth markers
                if(world_points.size() >= 3 && world_points.size() == camera_points.size()){
                    // New Pose!
                    Eigen::Matrix<float,3,1> new_tr;
                    Eigen::Quaternion<float> new_q;
                    Eigen::Matrix<float,3,1> proj_error;

                    #ifdef DEBUG_HORN
                    // Calculate the Horn rigid transformation
                    std::cout << style::green << style::bold << " Calculating Horn Transform" << style::normal<< std::endl;
                    #endif

                    #ifdef TIME_HORN
                    local_t1 = hr_clock::now();
                    #endif
                    calculateHorn(translation, rotation);
                    #ifdef TIME_HORN
                    local_t2 = hr_clock::now();
                    time_pose_estimation = std::chrono::duration_cast<std::chrono::microseconds>( local_t2 - local_t1 ).count();
                    std::cout << "HORN Duration: " << time_pose_estimation << std::endl;
                    #endif

                    #ifdef DEBUG_HORN
                    kstPoseUnfilteredUpdate(translation, rotation);
                    #endif

                    // Calculate the Porjection Error using the and update estimated points
                    calculateProjError(translation, rotation, proj_error);

                    is_lost = false;
                    if(proj_error.coeffRef(0) > 0.25 || proj_error.coeffRef(1) > 0.25 || proj_error.coeffRef(2) > 0.25){
                        #ifdef DEBUG_HORN
                        std::cout << style::red << "     Maximum projection error reached"<< style::normal <<": "<<proj_error.transpose()<< std::endl;
                        #endif
                        is_lost = true;
                    }

                }
                else{
                    #ifdef DEBUG_HORN
                    std::cout << style::red<<style::bold << " Not enough points" << style::normal << std::endl;
                    #endif
                    is_lost = true;
                }

                if(world_box_points.size() >= 3 && world_box_points.size() == camera_box_points.size()){
                    // Calculate the Horn rigid transformation and send to KST result of calculation
                    #ifdef DEBUG_HORN
                    std::cout << style::green << style::bold << " Calculating Horn Transform for Box" << style::normal<< std::endl;
                    #endif
                    calculateBoxHorn(box_translation, box_rotation);
                }
                else{
                    #ifdef DEBUG_HORN
                    std::cout << style::red<<style::bold << " Not enough points" << style::normal << std::endl;
                    #endif
                    box_is_lost = true;
                }
            }
            else{
                is_lost = true;
                box_is_lost = true;
            }

            // TODO: Estimate the location of all world_markers in relation to the camera frame according to the current camera pose.
            // Used to remove outliers
            if(!is_lost){
                for( const auto& it : all_markers_world ){
                    if( 0 <= it.first && it.first <= 16 ){
                        estimated_markers_depth[it.first] = (rotation.inverse()._transformVector(it.second-translation))(0,0);
                    }
                    else if( 18 <= it.first && it.first <= 31 ){
                        estimated_markers_depth[it.first] = (box_rotation.inverse()._transformVector(it.second-box_translation))(0,0);
                    }
                }
            }


            // Send new Camera Link TF
            tf_camera_transform.translation.x = translation.coeffRef(0);
            tf_camera_transform.translation.y = translation.coeffRef(1);
            tf_camera_transform.translation.z = translation.coeffRef(2);

            // verify if necessary (inverse from the horn algorithm shouldnt denormalize the quaternion)
            rotation.normalize();
            tf_camera_transform.rotation.x = rotation.x();
            tf_camera_transform.rotation.y = rotation.y();
            tf_camera_transform.rotation.z = rotation.z();
            tf_camera_transform.rotation.w = rotation.w();

            tf_camera_link.header.stamp = rgb_msg->header.stamp;//ros::Time::now();
            tf_camera_link.transform = tf_camera_transform;
            tf_broadcaster.sendTransform(tf_camera_link);

            geometry_msgs::PoseStamped horn_pose_msg;
            horn_pose_msg.header = depth_msg->header;
            horn_pose_msg.pose.orientation = tf_camera_transform.rotation;
            horn_pose_msg.pose.position.x = tf_camera_transform.translation.x;
            horn_pose_msg.pose.position.y = tf_camera_transform.translation.y;
            horn_pose_msg.pose.position.z = tf_camera_transform.translation.z;
            horn_pose_pub.publish(horn_pose_msg);


            // CHANGED:
            tf_box_transform.translation.x = box_translation.coeffRef(0);
            tf_box_transform.translation.y = box_translation.coeffRef(1);
            tf_box_transform.translation.z = box_translation.coeffRef(2);

            // verify if necessary (inverse from the horn algorithm shouldnt denormalize the quaternion)
            box_rotation.normalize();
            tf_box_transform.rotation.x = box_rotation.x();
            tf_box_transform.rotation.y = box_rotation.y();
            tf_box_transform.rotation.z = box_rotation.z();
            tf_box_transform.rotation.w = box_rotation.w();

            tf_box_link.header.stamp = rgb_msg->header.stamp;
            tf_box_link.transform = tf_box_transform;
            tf_broadcaster.sendTransform(tf_box_link);

            geometry_msgs::PoseStamped box_pose_msg;
            box_pose_msg.header = depth_msg->header;
            box_pose_msg.pose.orientation = tf_box_transform.rotation;
            box_pose_msg.pose.position.x = tf_box_transform.translation.x;
            box_pose_msg.pose.position.y = tf_box_transform.translation.y;
            box_pose_msg.pose.position.z = tf_box_transform.translation.z;
            box_horn_pub.publish(box_pose_msg);

            // Markers messages, REMOVE IF NOT NECESSARY!!
            world_markers_msg.header.stamp = tf_camera_link.header.stamp;
            camera_markers_msg.header.stamp = tf_camera_link.header.stamp;
            world_markers_pub.publish(world_markers_msg);
            camera_markers_pub.publish(camera_markers_msg);
            camera_box_markers_msg.header.stamp = tf_box_link.header.stamp;
            camera_box_markers_pub.publish(camera_box_markers_msg);


            #ifdef TIME_HORN
            cb_t2 = hr_clock::now();
            time_callback = std::chrono::duration_cast<std::chrono::microseconds>( cb_t2 - cb_t1 ).count();
            std::cout << time_detect_markers << " : " << time_callback << std::endl;
            KstPose.add(tf_camera_link.header.stamp.toSec()-start_time, time_detect_markers, time_pose_estimation, tf_camera_transform);
            #endif

            #ifdef DEBUG_HORN
            // Show color image
            cv::imshow("rgb_image",markers_image);
            cv::waitKey(1);
            #endif
        }


//===========================================================================================================
//    #####                                                         #     #
//   #     #   ##   #       ####  #    # #        ##   ##### ###### #     #  ####  #####  #    #
//   #        #  #  #      #    # #    # #       #  #    #   #      #     # #    # #    # ##   #
//   #       #    # #      #      #    # #      #    #   #   #####  ####### #    # #    # # #  #
//   #       ###### #      #      #    # #      ######   #   #      #     # #    # #####  #  # #
//   #     # #    # #      #    # #    # #      #    #   #   #      #     # #    # #   #  #   ##
//    #####  #    # ######  ####   ####  ###### #    #   #   ###### #     #  ####  #    # #    #
//===========================================================================================================
        void calculateHorn(Eigen::Matrix<float,3,1>& tr, Eigen::Quaternion<float>& q){
// Stats:
//  ~ 30 microseconds

            // 1. Calculate the centroid for each set of points
            camera_centroid /= world_points.size();
            world_centroid /= world_points.size();
            Eigen::Matrix<float,3,3> S = Eigen::Matrix<float,3,3>::Zero();
            Eigen::Matrix<float,4,4> N;

            for(std::size_t ii=0; ii<world_points.size(); ++ii){
                // 2. Calculate the error vectors
                camera_vectors[ii] = camera_centroid - camera_points[ii];
                world_vectors[ii] = world_centroid - world_points[ii];

                // 3. & 4. Calculate the covariance matrix
                S += camera_vectors[ii]*world_vectors[ii].transpose();
            }

            // 5. Construct the N matrix
            N.coeffRef(0,0) =  S.coeffRef(0,0) + S.coeffRef(1,1) + S.coeffRef(2,2);
            N.coeffRef(1,1) =  S.coeffRef(0,0) - S.coeffRef(1,1) - S.coeffRef(2,2);
            N.coeffRef(2,2) = -S.coeffRef(0,0) + S.coeffRef(1,1) - S.coeffRef(2,2);
            N.coeffRef(3,3) = -S.coeffRef(0,0) - S.coeffRef(1,1) + S.coeffRef(2,2);
            N.coeffRef(0,1) = N.coeffRef(1,0) = S.coeffRef(1,2) - S.coeffRef(2,1);
            N.coeffRef(0,2) = N.coeffRef(2,0) = S.coeffRef(2,0) - S.coeffRef(0,2);
            N.coeffRef(0,3) = N.coeffRef(3,0) = S.coeffRef(0,1) - S.coeffRef(1,0);
            N.coeffRef(2,1) = N.coeffRef(1,2) = S.coeffRef(0,1) + S.coeffRef(1,0);
            N.coeffRef(3,1) = N.coeffRef(1,3) = S.coeffRef(2,0) + S.coeffRef(0,2);
            N.coeffRef(3,2) = N.coeffRef(2,3) = S.coeffRef(1,2) + S.coeffRef(2,1);

            // 6. Solve the Eigen Values!
            eig_solver.compute(N,true);
            Eigen::Matrix<std::complex<float>,4,1> eig_values = eig_solver.eigenvalues();
            Eigen::Matrix<std::complex<float>,4,4> eig_vectors = eig_solver.eigenvectors();

            float max_value = -99999999.0;
            int max_idx = 0;
            for(int ii=0; ii<4; ++ii){
                if(eig_values.coeffRef(ii).real() > max_value){
                    max_value = eig_values.coeffRef(ii).real();
                    max_idx = ii;
                }
            }

            q.w() = eig_vectors(0,max_idx).real();
            q.x() = eig_vectors(1,max_idx).real();
            q.y() = eig_vectors(2,max_idx).real();
            q.z() = eig_vectors(3,max_idx).real();
            q.normalize();
            if(q.w() < 0){
                q.w() *= -1;
                q.vec() *= -1;
            }

            // 7. Calculate the Scale
            float camera_squared_sum = 0;
            float world_squared_sum = 0;
            for(std::size_t ii=0; ii < world_points.size(); ++ii){
                camera_squared_sum += ((camera_vectors[ii].coeffRef(0)*camera_vectors[ii].coeffRef(0))
                    + (camera_vectors[ii].coeffRef(1)*camera_vectors[ii].coeffRef(1))
                    + (camera_vectors[ii].coeffRef(2)*camera_vectors[ii].coeffRef(2)));

                world_squared_sum += ((world_vectors[ii].coeffRef(0)*world_vectors[ii].coeffRef(0))
                    + (world_vectors[ii].coeffRef(1)*world_vectors[ii].coeffRef(1))
                    + (world_vectors[ii].coeffRef(2)*world_vectors[ii].coeffRef(2)));
            }
            scale = 1; // CHANGED: Verify scale value result, remove computation is possible

            // 8. Calculate the translation!!
            tr = world_centroid - scale*q._transformVector(camera_centroid);


        }

//===========================================================================================================
//    #####                                                         #     #
//   #     #   ##   #       ####  #    # #        ##   ##### ###### #     #  ####  #####  #    #
//   #        #  #  #      #    # #    # #       #  #    #   #      #     # #    # #    # ##   #
//   #       #    # #      #      #    # #      #    #   #   #####  ####### #    # #    # # #  #
//   #       ###### #      #      #    # #      ######   #   #      #     # #    # #####  #  # #
//   #     # #    # #      #    # #    # #      #    #   #   #      #     # #    # #   #  #   ##
//    #####  #    # ######  ####   ####  ###### #    #   #   ###### #     #  ####  #    # #    #
//===========================================================================================================
        void calculateBoxHorn(Eigen::Matrix<float,3,1>& tr, Eigen::Quaternion<float>& q){
// Stats:
//  ~ 30 microseconds

            #ifdef TIME_HORN
            std::chrono::high_resolution_clock::time_point cb_t1, cb_t2;
            cb_t1 = hr_clock::now();
            #endif
            // 1. Calculate the centroid for each set of points
            camera_box_centroid /= world_box_points.size();
            world_box_centroid /= world_box_points.size();
            Eigen::Matrix<float,3,3> S = Eigen::Matrix<float,3,3>::Zero();
            Eigen::Matrix<float,4,4> N;

            for(std::size_t ii=0; ii<world_box_points.size(); ++ii){
                // 2. Calculate the error vectors
                camera_box_vectors[ii] = camera_box_centroid - camera_box_points[ii];
                world_box_vectors[ii] = world_box_centroid - world_box_points[ii];

                // 3. & 4. Calculate the covariance matrix
                S += camera_box_vectors[ii]*world_box_vectors[ii].transpose();
            }

            // 5. Construct the N matrix
            N.coeffRef(0,0) =  S.coeffRef(0,0) + S.coeffRef(1,1) + S.coeffRef(2,2);
            N.coeffRef(1,1) =  S.coeffRef(0,0) - S.coeffRef(1,1) - S.coeffRef(2,2);
            N.coeffRef(2,2) = -S.coeffRef(0,0) + S.coeffRef(1,1) - S.coeffRef(2,2);
            N.coeffRef(3,3) = -S.coeffRef(0,0) - S.coeffRef(1,1) + S.coeffRef(2,2);
            N.coeffRef(0,1) = N.coeffRef(1,0) = S.coeffRef(1,2) - S.coeffRef(2,1);
            N.coeffRef(0,2) = N.coeffRef(2,0) = S.coeffRef(2,0) - S.coeffRef(0,2);
            N.coeffRef(0,3) = N.coeffRef(3,0) = S.coeffRef(0,1) - S.coeffRef(1,0);
            N.coeffRef(2,1) = N.coeffRef(1,2) = S.coeffRef(0,1) + S.coeffRef(1,0);
            N.coeffRef(3,1) = N.coeffRef(1,3) = S.coeffRef(2,0) + S.coeffRef(0,2);
            N.coeffRef(3,2) = N.coeffRef(2,3) = S.coeffRef(1,2) + S.coeffRef(2,1);

            // 6. Solve the Eigen Values!
            eig_solver.compute(N,true);
            Eigen::Matrix<std::complex<float>,4,1> eig_values = eig_solver.eigenvalues();
            Eigen::Matrix<std::complex<float>,4,4> eig_vectors = eig_solver.eigenvectors();

            float max_value = -99999999.0;
            int max_idx = 0;
            for(int ii=0; ii<4; ++ii){
                if(eig_values.coeffRef(ii).real() > max_value){
                    max_value = eig_values.coeffRef(ii).real();
                    max_idx = ii;
                }
            }

            q.w() = eig_vectors(0,max_idx).real();
            q.x() = eig_vectors(1,max_idx).real();
            q.y() = eig_vectors(2,max_idx).real();
            q.z() = eig_vectors(3,max_idx).real();
            q.normalize();
            q = q.inverse();
            if(q.w() < 0){
                q.w() *= -1;
                q.vec() *= -1;
            }

            // 7. Calculate the Scale
            float camera_squared_sum = 0;
            float world_squared_sum = 0;
            for(std::size_t ii=0; ii < world_box_points.size(); ++ii){
                camera_squared_sum += ((camera_box_vectors[ii].coeffRef(0)*camera_box_vectors[ii].coeffRef(0))
                    + (camera_box_vectors[ii].coeffRef(1)*camera_box_vectors[ii].coeffRef(1))
                    + (camera_box_vectors[ii].coeffRef(2)*camera_box_vectors[ii].coeffRef(2)));

                world_squared_sum += ((world_box_vectors[ii].coeffRef(0)*world_box_vectors[ii].coeffRef(0))
                    + (world_box_vectors[ii].coeffRef(1)*world_box_vectors[ii].coeffRef(1))
                    + (world_box_vectors[ii].coeffRef(2)*world_box_vectors[ii].coeffRef(2)));
            }
            scale = 1; // CHANGED: verify scale value!! Remove if not necessary

            // 8. Calculate the translation!!
            tr = camera_box_centroid - scale*q._transformVector(world_box_centroid);

            #ifdef TIME_HORN
            cb_t2 = std::chrono::high_resolution_clock::now();
            auto cb_duration = std::chrono::duration_cast<std::chrono::microseconds>( cb_t2 - cb_t1 ).count();
            std::cout << cb_duration << " : Complete callback" << std::endl;
            #endif
        }

//===========================================================================================================
//    #####                                                         ######                       #######
//   #     #   ##   #       ####  #    # #        ##   ##### ###### #     # #####   ####       # #       #####  #####   ####  #####
//   #        #  #  #      #    # #    # #       #  #    #   #      #     # #    # #    #      # #       #    # #    # #    # #    #
//   #       #    # #      #      #    # #      #    #   #   #####  ######  #    # #    #      # #####   #    # #    # #    # #    #
//   #       ###### #      #      #    # #      ######   #   #      #       #####  #    #      # #       #####  #####  #    # #####
//   #     # #    # #      #    # #    # #      #    #   #   #      #       #   #  #    # #    # #       #   #  #   #  #    # #   #
//    #####  #    # ######  ####   ####  ###### #    #   #   ###### #       #    #  ####   ####  ####### #    # #    #  ####  #    #
//===========================================================================================================
        void calculateProjError(Eigen::Matrix<float,3,1>& tr, Eigen::Quaternion<float>& q, Eigen::Matrix<float,3,1>& proj_error){
            Eigen::Matrix<float,3,1> proj_point;
            proj_error.setZero();

            for( int ii=0; ii < world_points.size(); ++ii){
                proj_point = world_points[ii]-(tr+q._transformVector(camera_points[ii]));

                proj_error.coeffRef(0) += (proj_point.coeffRef(0)*proj_point.coeffRef(0));
                proj_error.coeffRef(1) += (proj_point.coeffRef(1)*proj_point.coeffRef(1));
                proj_error.coeffRef(2) += (proj_point.coeffRef(2)*proj_point.coeffRef(2));
            }

            proj_error /= world_points.size();
        }


//===========================================================================================================
//                    #    #
//   # #    # # ##### #   #   ####  #####
//   # ##   # #   #   #  #   #        #
//   # # #  # #   #   ###     ####    #
//   # #  # # #   #   #  #        #   #
//   # #   ## #   #   #   #  #    #   #
//   # #    # #   #   #    #  ####    #
//===========================================================================================================
        void initKst(){

            kstPoseUnfilteredUpdate(translation,rotation);
            kstPoseFilteredUpdate(translation,rotation);
            std::cout << LOG_ID("HornTransform") << "KST initialization complete.";
        }

//===========================================================================================================
//   #    #              ######                       #     #
//   #   #   ####  ##### #     #  ####   ####  ###### #     # #####  #####    ##   ##### ######
//   #  #   #        #   #     # #    # #      #      #     # #    # #    #  #  #    #   #
//   ###     ####    #   ######  #    #  ####  #####  #     # #    # #    # #    #   #   #####
//   #  #        #   #   #       #    #      # #      #     # #####  #    # ######   #   #
//   #   #  #    #   #   #       #    # #    # #      #     # #      #    # #    #   #   #
//   #    #  ####    #   #        ####   ####  ######  #####  #      #####  #    #   #   ######
//===========================================================================================================
        void kstPoseUnfilteredUpdate(const Eigen::Matrix<float,3,1>& new_tr, const Eigen::Quaternion<float>& new_q){
            static bool first_time = true;
            static bool second_time = true;
            static std::ofstream kst_pose;
            static Eigen::Matrix<float,3,1> tr;
            static Eigen::Quaternion<float> q;

            // NOTE: The chosen architecture for the update function, makes it easy to have the same synergy? between
            // the First Row with the names and the necessary adjustments in case of change
            // The necessary code is all in one place!

            if( first_time ){
                // First time running the function so open the corresponding file and print variable names in first row
                // Logging for KST
                kst_pose.open(save_dir+"kst/pose_unfiltered.csv", std::ofstream::out);
                kst_pose  << "time,"
                            << "x,"
                            << "y,"
                            << "z,"
                            << "qx,"
                            << "qy,"
                            << "qz,"
                            << "qw" << std::endl;
                if( kst_pose.fail() ){
                    std::cerr << "Failed to open [" << save_dir << "kst/pose_unfiltered.csv] file, try again next time." << std::endl;
                }
                else{
                    first_time = false;
                }

                tr.setZero();
                q.vec().setZero();
                q.w() = 1;
            }
            else{
                if(second_time){
                    tr = new_tr;
                    q = new_q;
                    second_time = false;
                }
                // The rest of the times just do normal line print
                kst_pose  << current_time-0.0001 << ","
                            << tr.coeffRef(0) << ","
                            << tr.coeffRef(1) << ","
                            << tr.coeffRef(2) << ","
                            << q.x() << ","
                            << q.y() << ","
                            << q.z() << ","
                            << q.w() << std::endl;
                // The rest of the times just do normal line print
                kst_pose  << current_time << ","
                            << new_tr.coeffRef(0) << ","
                            << new_tr.coeffRef(1) << ","
                            << new_tr.coeffRef(2) << ","
                            << new_q.x() << ","
                            << new_q.y() << ","
                            << new_q.z() << ","
                            << new_q.w() << std::endl;
                tr = new_tr;
                q = new_q;
            }

        }

        void kstPoseFilteredUpdate(const Eigen::Matrix<float,3,1>& new_tr, const Eigen::Quaternion<float>& new_q){
            static bool first_time = true;
            static bool second_time = true;
            static std::ofstream kst_pose;
            static Eigen::Matrix<float,3,1> tr;
            static Eigen::Quaternion<float> q;

            // NOTE: The chosen architecture for the update function, makes it easy to have the same synergy? between
            // the First Row with the names and the necessary adjustments in case of change
            // The necessary code is all in one place!

            if( first_time ){
                // First time running the function so open the corresponding file and print variable names in first row
                // Logging for KST
                kst_pose.open(save_dir+"kst/pose_filtered.csv", std::ofstream::out);
                kst_pose  << "time,"
                            << "x,"
                            << "y,"
                            << "z,"
                            << "qx,"
                            << "qy,"
                            << "qz,"
                            << "qw" << std::endl;
                if( kst_pose.fail() ){
                    std::cerr << "Failed to open [" << save_dir << "kst/pose_filtered.csv] file, try again next time." << std::endl;
                }
                else{
                    first_time = false;
                }

                tr.setZero();
                q.vec().setZero();
                q.w() = 1;
            }
            else{
                if(second_time){
                    tr = new_tr;
                    q = new_q;
                    second_time = false;
                }
                // The rest of the times just do normal line print
                kst_pose  << current_time-0.0001 << ","
                            << tr.coeffRef(0) << ","
                            << tr.coeffRef(1) << ","
                            << tr.coeffRef(2) << ","
                            << q.x() << ","
                            << q.y() << ","
                            << q.z() << ","
                            << q.w() << std::endl;
                // The rest of the times just do normal line print
                kst_pose  << current_time << ","
                            << new_tr.coeffRef(0) << ","
                            << new_tr.coeffRef(1) << ","
                            << new_tr.coeffRef(2) << ","
                            << new_q.x() << ","
                            << new_q.y() << ","
                            << new_q.z() << ","
                            << new_q.w() << std::endl;
                tr = new_tr;
                q = new_q;
            }

        }


    };

    PLUGINLIB_DECLARE_CLASS(pupil_utils, HornTransform, pupil_utils::HornTransform, nodelet::Nodelet);

}
