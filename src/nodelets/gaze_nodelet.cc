/**
@author     johnmper
@file       gaze_nodelet.cc
@brief      ROS nodelet responsable for the update of the Rviz TF for the Eye and Gaze 3D World Line
*/

// C++ STANDARD INCLUDES
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

// ROS INCLUDES
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

// ROS MESSAGES
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/JointState.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PointStamped.h>
#include <pupil_msgs/eye_status.h>
#include <pupil_msgs/point2d.h>
#include <kinova_msgs/GoToPose.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

// OTHER INCLUDES
#include <Eigen/Eigen>
#include <SFML/Audio.hpp>

// LOCAL INCLUDES
#undef DEBUG
#include <ros_utils/debug.hh>
#include <ros_utils/math.hh>
#include <ros_utils/style.hh>
#include <ros_utils/filter.hh>
#include <ros_utils/beeps.hh>
#include <ros_utils/circular_array.hh>

// Defines the correct parameters for the Simulation
#undef GAZEBO_SIMULATION
#undef FIXED_MARKERS_POSE_GOAL

// Simulated Pupil Tracker Parameters
#ifdef GAZEBO_SIMULATION
#define DEPTH_WINDOW_SZ   7     // Depth Neighbor Size
#define FIXATION_BUF_SIZE 16    // 120fps -> 0.00833s   || Size: 72 == 0.6 seconds
#define HISTORY_BUF_SIZE  12    // 120fps -> 0.00833s   || Size: 72 == 0.6 seconds
#define CONFIDENCE_BUF_SIZE 4

#define MAX_BLINK_TM      0.5
#define MIN_BLINK_TM      0.035
#define MAX_DOUBLE_BLINK_TM     0.8

#define DISPERION_CONFIDENCE_THRESH 0.8
#define DISPERSION_THRESH       0.001
#define GAZE_CONFIDENCE_THRESH  0.8
#define GAZE_POSITION_THRESH    0.05
#define GAZE_ORIENTATION_THRESH 0.2

#else   // Real Pupil Glasses Parameters
#define DEPTH_WINDOW_SZ   22    // Depth Neighbor Size
#define FIXATION_BUF_SIZE 256   // 120fps -> 0.00833s   || Size: 72 == 0.6 seconds
#define HISTORY_BUF_SIZE  256    // 120fps -> 0.00833s   || Size: 72 == 0.6 seconds
#define CONFIDENCE_BUF_SIZE 24

#define MINIMUM_FIXATION_DURATION 2

#define MAX_BLINK_TM      0.5
#define MIN_BLINK_TM      0.035
#define MAX_DOUBLE_BLINK_TM     0.8

#define DISPERION_CONFIDENCE_THRESH 0.8
#define DISPERSION_THRESH       0.001
#define GAZE_CONFIDENCE_THRESH  0.8
#define GAZE_POSITION_THRESH    0.15
#define GAZE_ORIENTATION_THRESH 0.7
#endif

#undef TIME_GAZE

using hr_clock = std::chrono::high_resolution_clock;

#ifdef TIME_GAZE
#define MS_CAST(a,b) std::chrono::duration_cast<std::chrono::microseconds>(a-b)

std::chrono::high_resolution_clock::time_point t1,t2;
double debug_duration=0.0;
#define TIME_FUNCTION_DEBUG(a) \
    t1 = std::chrono::high_resolution_clock::now();\
    a;\
    t2 = std::chrono::high_resolution_clock::now();\
    debug_duration = (double)std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();\
    std::cout << "Took: " << debug_duration << " us" << std::endl;
#endif



void printPose(const geometry_msgs::Pose& pose){
    std::cout << "{{"<<pose.position.x<<", "<<pose.position.y<<", "<<pose.position.z <<"},";
    std::cout << "{" << pose.orientation.w<<", "<<pose.orientation.x<<", "<<pose.orientation.y<<", "<<pose.orientation.z<<"}}";
}

struct Pose3D
{
    Eigen::Matrix<double,3,1> position;
    Eigen::Quaternion<double> orientation;

    Pose3D(){
        position << 0,0,0;
        orientation.w() = 1;
        orientation.vec() << 0, 0, 0;
    }

    Pose3D(double _x,double _y, double _z, double _qw, double _qx, double _qy, double _qz){
        position << _x, _y, _z;
        orientation.w() = _qw;
        orientation.vec() << _qx, _qy, _qz;
    }


    Pose3D(geometry_msgs::Pose _pose){
        position << _pose.position.x, _pose.position.y, _pose.position.z;
        orientation.w() = _pose.orientation.w;
        orientation.vec() << _pose.orientation.x, _pose.orientation.y, _pose.orientation.z;
    }

    geometry_msgs::Pose toRosPose(){
        geometry_msgs::Pose _pose;
        _pose.position.x = position.coeffRef(0);
        _pose.position.y = position.coeffRef(1);
        _pose.position.z = position.coeffRef(2);
        _pose.orientation.w = orientation.w();
        _pose.orientation.x = orientation.x();
        _pose.orientation.y = orientation.y();
        _pose.orientation.z = orientation.z();
        return _pose;
    }

    double& x(){
        return position.coeffRef(0);
    }
    double& y(){
        return position.coeffRef(1);
    }
    double& z(){
        return position.coeffRef(2);
    }

};

struct KstGazeGoal{
    std::string filepath;
    std::ofstream file;

    KstGazeGoal(std::string filepath_) : filepath(filepath_){
        file.open(filepath,std::ofstream::out);
        if(file.fail()){
            std::cerr << "Failed to create " << filepath << " file." << std::endl;
        }else{
            file << "time, dispersion, tx, ty, tz, qx, qy, qz, qw";
            file << std::endl;
        }
    }
    ~KstGazeGoal(){
        file.close();
    }

    bool add(double tm,double dispersion, geometry_msgs::Pose goal_ ){

        if(!file.is_open()){
            std::cerr << "File isn't opened or was prematurely closed" << std::endl;
            return false;
        }

        file << tm << ", ";
        file << dispersion << ", ";
        file << goal_.position.x << ", " << goal_.position.y << ", " << goal_.position.z << ", ";
        file << goal_.orientation.x << ", " << goal_.orientation.y << ", " << goal_.orientation.z << ", " << goal_.orientation.w;
        file << std::endl;

        return true;
    }
};


struct KstConfidence{
    std::string filepath;
    std::ofstream file;

    KstConfidence(std::string filepath_) : filepath(filepath_){
        file.open(filepath,std::ofstream::out);
        if(file.fail()){
            std::cerr << "Failed to create " << filepath << " file." << std::endl;
        }else{
            file << "time, mean_confidence, var_confidence, blink, fixation";
            file << std::endl;
        }
    }
    ~KstConfidence(){
        file.close();
    }

    bool add(double tm, double mean, double var, bool blink, bool fixation ){

        if(!file.is_open()){
            std::cerr << "File isn't opened or was prematurely closed" << std::endl;
            return false;
        }

        file << tm << ", ";
        file << mean << ", ";
        file << var << ", ";
        if( blink )
            file << 1 << ", ";
        else
            file << 0 << ", ";
        if( fixation )
            file << 1;
        else
            file << 0;
        file << std::endl;

        return true;
    }
};

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

    struct GazeData{
    	double tm;
    	double confidence;
    	geometry_msgs::Pose pose;

    	GazeData() : tm(0),confidence(0){}
    	GazeData(double tm_, double confidence_, geometry_msgs::Pose pose_) : tm(tm_), confidence(confidence_),pose(pose_){}

        void print(){
            std::cout << "{";
            std::cout << tm <<", ";
            std::cout << confidence << ", ";
            printPose(pose);
            std::cout << "}" <<std::endl;
        }
    };

    struct Gaze2D{
        double tm;
        double confidence;
        pupil_msgs::point2d point;

        Gaze2D() : tm(0),confidence(0){}
        Gaze2D(double tm_, double confidence_, pupil_msgs::point2d point_) : tm(tm_), confidence(confidence_), point(point_){}
        Gaze2D(const Gaze2D& gaze) : tm(gaze.tm), confidence(gaze.confidence), point(gaze.point){}
        void print() const {
            std::cout << "{" << tm << ", " << confidence << ", {" << point.x << ", "<<point.y<<"}}";
        }
    };

//===========================================================================================================
//       #####                       #     #
//      #     #   ##   ###### ###### ##    #  ####  #####  ###### #      ###### #####
//      #        #  #      #  #      # #   # #    # #    # #      #      #        #
//      #  #### #    #    #   #####  #  #  # #    # #    # #####  #      #####    #
//      #     # ######   #    #      #   # # #    # #    # #      #      #        #
//      #     # #    #  #     #      #    ## #    # #    # #      #      #        #
//       #####  #    # ###### ###### #     #  ####  #####  ###### ###### ######   #
//===========================================================================================================
    class GazeNodelet: public nodelet::Nodelet{
    private:

        ros::ServiceClient goal_srv;
        /// Gaze subscriber
        ros::Subscriber eye_status_sub;
        /// Horn pose callback
        ros::Subscriber horn_pose_sub;
        /// Box Horn pose callback
        ros::Subscriber box_pose_sub;
        /// Subscriber for the pointcloud callback
        ros::Subscriber pc_sub;
        /// Subscriber with filter for the depth image
        image_transport::Subscriber depth_sub;
        /// Last depth received Image
        sensor_msgs::Image last_depth;
        /// Estimated 3D Gaze Point
        ros::Publisher gaze_3d_pub;
        /// Estimated plane pose publisher
        ros::Publisher plane_pose_pub;
        /// Estimated plane pose publisher
        ros::Publisher mean_pose_pub;
        /// Estimated plane pose publisher
        ros::Publisher goal_pose_pub;

        /// 3D desired Point
        ros::Publisher point_marker_pub;
        /// Eye Marker Publisher
        ros::Publisher eye_marker_pub;
        /// Gaze Line Marker Publisher
        ros::Publisher gaze_line_marker_pub;

        /// Processed Gaze Message
        geometry_msgs::PointStamped gaze_3d_msg;
        /// 3D points window, for plane estimation
        std::vector<geometry_msgs::Point> plane_3d_points;
        /// 3D estimated Pose of the tangential plane of the point cloud in the line gaze interesection point
        geometry_msgs::PoseStamped plane_pose;
        /// 3D estimated Pose of the tangential plane of the point cloud in the line gaze interesection point
        geometry_msgs::PoseStamped mean_pose;
        /// Goal Pose, extracted from eye blink
        geometry_msgs::PoseStamped goal_pose;

        /// Rviz 3d desired point  Marker message
        visualization_msgs::Marker point_marker_msg;
        /// Rviz Eye Center Marker message
        visualization_msgs::Marker eye_marker;
        /// Rviz Eye Center Marker message
        visualization_msgs::Marker gaze_line_marker;

        /// [camera_link -> eye] transform publisher
        tf::TransformBroadcaster tf_broadcaster;
        /// Smaller piece of the Transform message
        geometry_msgs::Transform tf_eye_transform;
        /// Actual [camera_link -> eye] transform message
        geometry_msgs::TransformStamped tf_eye;
        /// WORLD intrinsics focal length X
        double fx;
        /// WORLD intrinsics focal length Y
        double fy;
        /// WORLD intrinsics image center X
        double cx;
        /// WORLD intrinsics image center Y
        double cy;
        /// Eigen value and vetors solver
        Eigen::EigenSolver<Eigen::Matrix<double,3,3>> eig_solver;


        bool is_lost;
        bool depth_init_completed;
        /// Utility structure for KST plotting
        KstGazeGoal kst;
        KstGazeGoal kst_goal;
        KstConfidence kst_confidence;
        double start_time;

        audio::Beeps<780> beep;

        /// Gaze History array
        CircularArray<GazeData,HISTORY_BUF_SIZE>   gaze_history;
        bool in_fixation;

        ros::Time last_tf_time;
        tf::TransformListener listener;

        geometry_msgs::Pose horn_pose;
        geometry_msgs::PoseStamped box_pose;


        /// World 3D location of the ARUCO markers
        std::vector<Pose3D> all_markers_world;
        std::vector<Pose3D> all_objects_world;


        // CHANGED: Added to work with the pointcloud instead of the depth image
        sensor_msgs::PointCloud2 pointcloud;
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
        GazeNodelet() : kst_goal("/home/johnmper/.ROSData/pupil/kst/goal_pose.txt"), kst("/home/johnmper/.ROSData/pupil/kst/gaze_goal.txt"), kst_confidence("/home/johnmper/.ROSData/pupil/kst/status.txt"),depth_init_completed(false),is_lost(true),in_fixation(false){


            // Initiate the markers messages for Rviz
            point_marker_msg.header.seq = 0;
            point_marker_msg.header.frame_id = "rgb_optical_frame";
            point_marker_msg.ns = "gaze_markers";
            point_marker_msg.id = 0;
            point_marker_msg.type = visualization_msgs::Marker::SPHERE;
            point_marker_msg.action = visualization_msgs::Marker::ADD;
            point_marker_msg.pose.orientation.x = 0;
            point_marker_msg.pose.orientation.y = 0;
            point_marker_msg.pose.orientation.z = 0;
            point_marker_msg.pose.orientation.w = 1;
            point_marker_msg.pose.position.x = 0;
            point_marker_msg.pose.position.y = 0;
            point_marker_msg.pose.position.z = 0;
            point_marker_msg.scale.x = 0.04;
            point_marker_msg.scale.y = 0.04;
            point_marker_msg.scale.z = 0.04;
            point_marker_msg.color.r = 1;
            point_marker_msg.color.g = 0;
            point_marker_msg.color.b = 0;
            point_marker_msg.color.a = 1;

            // Initiate the markers messages for Rviz
            eye_marker.header.seq = 0;
            eye_marker.header.frame_id = "rgb_optical_frame";
            eye_marker.ns = "gaze_markers";
            eye_marker.id = 0;
            eye_marker.type = visualization_msgs::Marker::SPHERE;
            eye_marker.action = visualization_msgs::Marker::ADD;
            eye_marker.pose.orientation.x = 0;
            eye_marker.pose.orientation.y = 0;
            eye_marker.pose.orientation.z = 0;
            eye_marker.pose.orientation.w = 1;
            eye_marker.pose.position.x = 0;
            eye_marker.pose.position.y = 0;
            eye_marker.pose.position.z = 0;
            eye_marker.scale.x = 0.04;
            eye_marker.scale.y = 0.04;
            eye_marker.scale.z = 0.04;
            eye_marker.color.r = 0.7;
            eye_marker.color.g = 0.7;
            eye_marker.color.b = 0.7;
            eye_marker.color.a = 1;

            // Initiate the markers messages for Rviz
            gaze_line_marker.header.seq = 0;
            gaze_line_marker.header.frame_id = "rgb_optical_frame";
            gaze_line_marker.ns = "gaze_markers";
            gaze_line_marker.id = 0;
            gaze_line_marker.type = visualization_msgs::Marker::ARROW;
            gaze_line_marker.action = visualization_msgs::Marker::ADD;
            gaze_line_marker.pose.orientation.x = 0;
            gaze_line_marker.pose.orientation.y = 0;
            gaze_line_marker.pose.orientation.z = 0;
            gaze_line_marker.pose.orientation.w = 1;
            gaze_line_marker.pose.position.x = 0;
            gaze_line_marker.pose.position.y = 0;
            gaze_line_marker.pose.position.z = 0;
            gaze_line_marker.scale.x = 0.005;
            gaze_line_marker.scale.y = 0.01;
            gaze_line_marker.scale.z = 0;
            gaze_line_marker.color.r = 1;
            gaze_line_marker.color.g = 0;
            gaze_line_marker.color.b = 0;
            gaze_line_marker.color.a = 1;
            gaze_line_marker.points.resize(2);

            // Populate the gaze-3d_msgs with initial values
            gaze_3d_msg.header.stamp = ros::Time::now();
            gaze_3d_msg.header.frame_id = "rgb_optical_frame";
            gaze_3d_msg.point.x = 0;
            gaze_3d_msg.point.y = 0;
            gaze_3d_msg.point.z = 0;

            // Populate the plane_pose message
            plane_pose.header.seq = 0;
            plane_pose.header.stamp = gaze_3d_msg.header.stamp;
            plane_pose.header.frame_id = "rgb_optical_frame";
            plane_pose.pose.position.x = 0;
            plane_pose.pose.position.y = 0;
            plane_pose.pose.position.z = 0;
            plane_pose.pose.orientation.x = 0;
            plane_pose.pose.orientation.y = 0;
            plane_pose.pose.orientation.z = 0;
            plane_pose.pose.orientation.w = 1;

            // Populate the plane_pose message
            goal_pose.header.seq = 0;
            goal_pose.header.stamp = gaze_3d_msg.header.stamp;
            goal_pose.header.frame_id = "rgb_optical_frame";
            goal_pose.pose.position.x = 0;
            goal_pose.pose.position.y = 0;
            goal_pose.pose.position.z = 0;
            goal_pose.pose.orientation.x = 0;
            goal_pose.pose.orientation.y = 0;
            goal_pose.pose.orientation.z = 0;
            goal_pose.pose.orientation.w = 1;

            horn_pose.position.x = 0;
            horn_pose.position.y = 0;
            horn_pose.position.z = 0;
            horn_pose.orientation.x = 0;
            horn_pose.orientation.y = 0;
            horn_pose.orientation.z = 0;
            horn_pose.orientation.w = 1;

            box_pose.header.seq = 0;
            box_pose.header.stamp = gaze_3d_msg.header.stamp;
            box_pose.header.frame_id = "rgb_optical_frame";
            box_pose.pose.position.x = 0;
            box_pose.pose.position.y = 0;
            box_pose.pose.position.z = 0;
            box_pose.pose.orientation.x = 0;
            box_pose.pose.orientation.y = 0;
            box_pose.pose.orientation.z = 0;
            box_pose.pose.orientation.w = 1;

            // Reserve enough memory for depth points
            plane_3d_points.reserve(DEPTH_WINDOW_SZ*DEPTH_WINDOW_SZ);

            // Reserve enough space for the objects in the world
            all_objects_world.reserve(16);
            all_markers_world.reserve(16);
            all_markers_world.push_back(Pose3D(-0.70,     0,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D(-0.70, -0.30,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D(-0.70, -0.60,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D(-0.35,     0,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D(-0.35, -0.30,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D(-0.35, -0.60,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D(  0.0, -0.30,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D(  0.0, -0.60,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D( 0.35,     0,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D( 0.35, -0.30,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D( 0.35, -0.60,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D( 0.70,     0,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D( 0.70, -0.30,0, 1, 0, 0, 0));
            all_markers_world.push_back(Pose3D( 0.70, -0.60,0, 1, 0, 0, 0));


            // D: [-0.08651317656040192, 0.06496115773916245, -0.0017410318832844496, 0.0022893715649843216, 0.0]
            fx = 1413.7486572265625;
            fy = 1416.033935546875;
            cx = 1020.4573364257812;
            cy = 518.3480859375;

            std::cout << LOG_ID("GazeNodelet") << "Created" << std::endl;
        }

        virtual ~GazeNodelet(){
        }


//===========================================================================================================
//   #######        ###
//   #     # #    #  #  #    # # #####
//   #     # ##   #  #  ##   # #   #
//   #     # # #  #  #  # #  # #   #
//   #     # #  # #  #  #  # # #   #
//   #     # #   ##  #  #   ## #   #
//   ####### #    # ### #    # #   #
//===========================================================================================================
    private:
        virtual void onInit(){

            ros::NodeHandle nh = getNodeHandle();
            /// Image transport for depth image
            image_transport::ImageTransport depth_it(nh);

            point_marker_pub = nh.advertise<visualization_msgs::Marker>("pupil/gaze_0/point_marker",0);
            eye_marker_pub = nh.advertise<visualization_msgs::Marker>("pupil/gaze_0/eye_marker",0);
            gaze_line_marker_pub = nh.advertise<visualization_msgs::Marker>("pupil/gaze_0/gazeline_marker",0);

            gaze_3d_pub = nh.advertise<geometry_msgs::PointStamped>("pupil/gaze_0/point",0);
            plane_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pupil/gaze_0/pose",0);
            mean_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pupil/gaze_0/mean",0);
            goal_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pupil/gaze_0/goal",0);

            goal_srv = nh.serviceClient<kinova_msgs::GoToPose>("/go_to_pose");


            //image_transport::TransportHints depth_hints("raw", ros::TransportHints(), getPrivateNodeHandle(), "depth_image_transport");
            //depth_sub = depth_it.subscribe("/pupil/depth_registered/image_raw",1,&pupil_utils::GazeNodelet::depthCallback,this,depth_hints);
            // CHANGED: Added to work with the pointcloud instead of the depth image
            pc_sub = nh.subscribe("/pupil/points",1,&pupil_utils::GazeNodelet::pointcloudCallback,this);
            eye_status_sub = nh.subscribe("/pupil/gaze_0/eye_status",1,&pupil_utils::GazeNodelet::statusCallback,this);
            horn_pose_sub = nh.subscribe("/horn/pose",1,&pupil_utils::GazeNodelet::hornCallback,this);
            box_pose_sub = nh.subscribe("/horn/box",1,&pupil_utils::GazeNodelet::boxCallback,this);

            start_time = ros::Time::now().toSec();
            std::cout << LOG_ID("GazeNodelet") << "Initialization Complete" << std::endl;
        }


//===========================================================================================================
//   ######                              #####
//   #     # ###### #####  ##### #    # #     #   ##   #      #      #####    ##    ####  #    #
//   #     # #      #    #   #   #    # #        #  #  #      #      #    #  #  #  #    # #   #
//   #     # #####  #    #   #   ###### #       #    # #      #      #####  #    # #      ####
//   #     # #      #####    #   #    # #       ###### #      #      #    # ###### #      #  #
//   #     # #      #        #   #    # #     # #    # #      #      #    # #    # #    # #   #
//   ######  ###### #        #   #    #  #####  #    # ###### ###### #####  #    #  ####  #    #
//===========================================================================================================
        void depthCallback(const sensor_msgs::ImageConstPtr& depth_msg){
            debug("depthCallback()");

            // Populate colored_msg data
            last_depth = *depth_msg;
            // uint16_t* T = ((uint16_t*)depth_msg->data.data());
            // for(int ii=0; ii<depth_msg->height; ++ii){
            //     uint16_t* H = T[ii*depth_msg->width];
            //     for(int jj=0; jj<depth_msg->width; ++jj){
            //         // H[jj]
            //     }
            // }
            depth_init_completed = true;
        }

        void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& pc_msg){
            // Store Pointcloud
            pointcloud = *pc_msg;

            depth_init_completed = true;
        }

        void hornCallback(const geometry_msgs::PoseStampedConstPtr& horn_msg ){
            debug("hornCallback");

            // Populate the r200 Camera pose in the World
            horn_pose = (horn_msg->pose);

        }

        void boxCallback(const geometry_msgs::PoseStampedConstPtr& box_msg ){
            debug("boxCallback");

            // Populate the r200 Camera pose in the World
            all_objects_world.clear();
            all_objects_world.push_back(Pose3D(box_msg->pose));

        }

//===========================================================================================================
//    #####                        #####
//   #     #   ##   ###### ###### #     #   ##   #      #      #####    ##    ####  #    #
//   #        #  #      #  #      #        #  #  #      #      #    #  #  #  #    # #   #
//   #  #### #    #    #   #####  #       #    # #      #      #####  #    # #      ####
//   #     # ######   #    #      #       ###### #      #      #    # ###### #      #  #
//   #     # #    #  #     #      #     # #    # #      #      #    # #    # #    # #   #
//    #####  #    # ###### ######  #####  #    # ###### ###### #####  #    #  ####  #    #
//===========================================================================================================
        void statusCallback(const pupil_msgs::eye_statusConstPtr& eye_status_msg){
            debug("----------------\ngazeCallback()");

            if( !depth_init_completed){
                debug("Initialization Not Completed");
                return;
            }
            debug("Initialization Completed");

            #ifdef TIME_GAZE
            std::chrono::high_resolution_clock::time_point cb_t1, cb_t2, local_t1, local_t2;;
            cb_t1 = hr_clock::now();
            #endif

            // Update message headers
            gaze_3d_msg.header.seq++;
            point_marker_msg.header.seq++;
            eye_marker.header.seq++;
            gaze_line_marker.header.seq++;
            plane_pose.header.seq++;
            gaze_3d_msg.header.stamp = eye_status_msg->header.stamp;
            point_marker_msg.header.stamp = gaze_3d_msg.header.stamp;
            eye_marker.header.stamp = gaze_3d_msg.header.stamp;
            gaze_line_marker.header.stamp = gaze_3d_msg.header.stamp;
            plane_pose.header.stamp = gaze_3d_msg.header.stamp;

            // Update Eye Marker message
            eye_marker.pose.position.x = eye_status_msg->eye_center_3d.x/1000.0;
            eye_marker.pose.position.y = eye_status_msg->eye_center_3d.y/1000.0;
            eye_marker.pose.position.z = eye_status_msg->eye_center_3d.z/1000.0;
            eye_marker_pub.publish(eye_marker);

            // Update Eye line Marker Message
            gaze_line_marker.points[0] = eye_marker.pose.position;
            gaze_line_marker.points[1].x = eye_status_msg->gaze_normal_3d.x*2.0;
            gaze_line_marker.points[1].y = eye_status_msg->gaze_normal_3d.y*2.0;
            gaze_line_marker.points[1].z = eye_status_msg->gaze_normal_3d.z*2.0;
            gaze_line_marker.color.r = (1-eye_status_msg->confidence);
            gaze_line_marker.color.g = eye_status_msg->confidence;
            gaze_line_marker.color.b = 0;
            gaze_line_marker_pub.publish(gaze_line_marker);

            // TEMP: Varibble for testing the confidence parameters
            geometry_msgs::Pose tmp;



            bool new_pose = false;
            // Uses the Gaze Datum information to extract the depth points in the vicinity of the gazeline interesection
            // with the pointcloud, only proceeds if it found 3 depths near each other
            if(!eye_status_msg->is_closed){
                // CHANGED: changed updateIntersectionPoint to updateIntersectionPoint2!!
                if(updateIntersectionPoint2(eye_status_msg)){
                    double pose_confidence = 0;
                    // Calculate the tangential plane to the point cloud in the intersection between the gazeline aDetectednd the point cloud
                    // NOTE: Uses the 3D Mean Point calculated in updateGazeDepthIntersectionPoints
                    if(calculatePlanePose(plane_pose.pose,pose_confidence)){
                        geometry_msgs::PoseStamped mean_gaze;
                        geometry_msgs::Pose var_gaze;

                        GazeData new_gaze(eye_status_msg->header.stamp.toSec()-start_time,eye_status_msg->confidence,plane_pose.pose);
                        gaze_history.add(new_gaze);

                        calculateMeanPose(mean_gaze.pose, gaze_history);
                        calculateVarPose(var_gaze, mean_gaze.pose, gaze_history);
                        double var_norm = normPositionVariance(var_gaze);
                        geometry_msgs::Pose goal_pose_;

                        // Ugly transformations
                        Eigen::Matrix<double,3,1> vec;
                        Pose3D goal_;
                        vec << mean_gaze.pose.position.x,mean_gaze.pose.position.y,mean_gaze.pose.position.z;
                        goal_.position << horn_pose.position.x, horn_pose.position.y, horn_pose.position.z;

                        Eigen::Quaternion<double> quat(mean_gaze.pose.orientation.w,mean_gaze.pose.orientation.x,mean_gaze.pose.orientation.y,mean_gaze.pose.orientation.z);
                        Eigen::Quaternion<double> horn_quat(horn_pose.orientation.w,horn_pose.orientation.x,horn_pose.orientation.y,horn_pose.orientation.z);
                        Eigen::Quaternion<double> r200_to_color_optical(0.5,-0.5,0.5,-0.5);
                        goal_.position += (horn_quat*r200_to_color_optical)._transformVector(vec);
                        goal_.orientation = (horn_quat*r200_to_color_optical*quat);
                        goal_.orientation.normalize();

                        mean_gaze.pose = goal_.toRosPose();
                        if( !std::isnan(mean_gaze.pose.position.x) && !std::isnan(mean_gaze.pose.orientation.w)){
                            kst.add(gaze_3d_msg.header.stamp.toSec()-start_time, var_norm, mean_gaze.pose);

                            #ifdef FIXED_MARKERS_POSE_GOAL
                                double sz_x = 0.15;
                                double sz_y = 0.15;
                                double sz_z = 0.15;
                                if(eye_status_msg->double_blink && acceptableVariance(var_gaze,true)){
                                    // geometry_msgs::Pose goal_pose_;
                                    //
                                    // // Ugly transformations
                                    // Eigen::Matrix<double,3,1> vec;
                                    // Pose3D goal_;
                                    // vec << mean_gaze.pose.position.x,mean_gaze.pose.position.y,mean_gaze.pose.position.z;
                                    // goal_.position << horn_pose.position.x, horn_pose.position.y, horn_pose.position.z;
                                    //
                                    // Eigen::Quaternion<double> quat(mean_gaze.pose.orientation.w,mean_gaze.pose.orientation.x,mean_gaze.pose.orientation.y,mean_gaze.pose.orientation.z);
                                    // Eigen::Quaternion<double> horn_quat(horn_pose.orientation.w,horn_pose.orientation.x,horn_pose.orientation.y,horn_pose.orientation.z);
                                    // Eigen::Quaternion<double> r200_to_color_optical(0.5,-0.5,0.5,-0.5);
                                    // goal_.position += (horn_quat*r200_to_color_optical)._transformVector(vec);
                                    // goal_.orientation = (horn_quat*r200_to_color_optical*quat);
                                    // goal_.orientation.normalize();
                                    //
                                    // mean_gaze.pose = goal_.toRosPose();

                                    // goal_quat and goal_tr represent mean_gase.pose in the jaco axis

                                    bool accepted_pose = false;
                                    for(auto& it : all_markers_world){
                                        if( it.x()-sz_x < mean_gaze.pose.position.x && mean_gaze.pose.position.x < it.x() +sz_x
                                        &&  it.y()-sz_y < mean_gaze.pose.position.y && mean_gaze.pose.position.y < it.y() +sz_y )
                                        {
                                            Eigen::Quaternion<double> rx(0,1,0,0);
                                            rx = it.orientation*rx;
                                            goal_pose_ = it.toRosPose();
                                            goal_pose_.orientation.w = rx.w();
                                            goal_pose_.orientation.x = rx.x();
                                            goal_pose_.orientation.y = rx.y();
                                            goal_pose_.orientation.z = rx.z();
                                            accepted_pose = true;
                                            break;
                                        }
                                    }


                                    for(auto& it : all_objects_world){
                                        Eigen::Quaternion<double> rx(0.7071,0.7071,0,0);
                                        Eigen::Quaternion<double> rz(0.7071,0,0,-0.7071);

                                        // std::cout << "-------------------";
                                        // std::cout << "\nGAZE: "; printPose(mean_gaze.pose);
                                        // std::cout << "\nBOX: "; printPose(it.toRosPose());

                                        Pose3D horn_(horn_pose.position.x, horn_pose.position.y, horn_pose.position.z,horn_pose.orientation.w,horn_pose.orientation.x,horn_pose.orientation.y,horn_pose.orientation.z);
                                        it.position = horn_.position + horn_.orientation._transformVector(it.position);
                                        it.orientation = (horn_.orientation*it.orientation);
                                        it.orientation.normalize();

                                        // std::cout << "\nHORN: "; printPose(horn_.toRosPose());
                                        // std::cout << "\nBOX: "; printPose(it.toRosPose());
                                        // std::cout << std::endl;

                                        Pose3D gaze_(mean_gaze.pose);
                                        Eigen::Matrix<double,3,1> diff;
                                        diff = it.orientation._transformVector(gaze_.position-it.position);

                                        if(-sz_x < diff.coeffRef(0) && diff.coeffRef(0) < sz_x
                                        && -sz_y < diff.coeffRef(1) && diff.coeffRef(1) < sz_y
                                        && -sz_z < diff.coeffRef(2) && diff.coeffRef(2) < sz_z )
                                        {
                                            it.orientation = it.orientation*rx*rz;
                                            goal_pose_ = it.toRosPose();
                                            accepted_pose = true;
                                            break;
                                        }
                                    }

                                    if(accepted_pose){
                                        publishGoal(goal_pose_,0.20);
                                    }
                                }
                            #else
                                if(eye_status_msg->double_blink){// && acceptableVariance(var_gaze)){
                                    publishGoal(mean_gaze.pose);
                                }
                            #endif
                            mean_gaze.header = plane_pose.header;
                            plane_pose_pub.publish(plane_pose);
                            mean_pose_pub.publish(mean_gaze);
                        }
                    }
                }

            }
            #ifdef TIME_GAZE
            cb_t2 = hr_clock::now();
            double time_callback = std::chrono::duration_cast<std::chrono::microseconds>( cb_t2 - cb_t1 ).count();
            std::cout << time_callback << std::endl;
            #endif

        }



//===========================================================================================================
//   #     #                                    #####                       ######
//   #     # #####  #####    ##   ##### ###### #     #   ##   ###### ###### #     # ###### #####  ##### #    #
//   #     # #    # #    #  #  #    #   #      #        #  #      #  #      #     # #      #    #   #   #    #
//   #     # #    # #    # #    #   #   #####  #  #### #    #    #   #####  #     # #####  #    #   #   ######
//   #     # #####  #    # ######   #   #      #     # ######   #    #      #     # #      #####    #   #    #
//   #     # #      #    # #    #   #   #      #     # #    #  #     #      #     # #      #        #   #    #
//    #####  #      #####  #    #   #   ######  #####  #    # ###### ###### ######  ###### #        #   #    #
//
//         ###                                                                       ######
//          #  #    # ##### ###### #####   ####  ######  ####  ##### #  ####  #    # #     #  ####  # #    # #####  ####
//          #  ##   #   #   #      #    # #      #      #    #   #   # #    # ##   # #     # #    # # ##   #   #   #
//          #  # #  #   #   #####  #    #  ####  #####  #        #   # #    # # #  # ######  #    # # # #  #   #    ####
//          #  #  # #   #   #      #####       # #      #        #   # #    # #  # # #       #    # # #  # #   #        #
//          #  #   ##   #   #      #   #  #    # #      #    #   #   # #    # #   ## #       #    # # #   ##   #   #    #
//         ### #    #   #   ###### #    #  ####  ######  ####    #   #  ####  #    # #        ####  # #    #   #    ####
//===========================================================================================================
        bool updateIntersectionPoint(const pupil_msgs::eye_statusConstPtr& eye_status_msg){
            debug("updateGazeDepthIntersectionPoints()");

            if(eye_status_msg->gaze_normal_3d.z > -0.0001 && 0.0001 > eye_status_msg->gaze_normal_3d.z){
                // Gaze Normal in Z is Zero, Invalid Gaze. SKIP!!
                return false;
            }

            // Calculate the intersection of the gazeline with the image plane
            int u = (0.5+cx+(eye_status_msg->gaze_normal_3d.x*fx)/eye_status_msg->gaze_normal_3d.z);
            int v = (0.5+cy+(eye_status_msg->gaze_normal_3d.y*fy)/eye_status_msg->gaze_normal_3d.z);

            // TODO: CHANGE (U,V) TO NORM_POSE!!!! ?? WTF HOW DOES IT WORKS LIKE THIS!?
            //std::cerr << "("<<u<<", "<<v<<")\n";

            // Verify that the previous normalized pixel values arent out of the bound of the image
            // Out of bounds so there wont be any valid depths, just SKIP!
            const unsigned int sz = DEPTH_WINDOW_SZ;
            const unsigned int step = last_depth.width;
            if( u <= sz || v <= sz || (int)last_depth.width-sz <= u || (int)last_depth.height-sz <= v ){
                std::cerr << style::yellow << "     Out of bounds error" << style::normal << std::endl;
                return false;
            }

            // Cleanup from previous iteration
            plane_3d_points.clear();
            // Get depth mean from the are surrounding the (u,v) coordiante in the depth_registered frame
            uint16_t* reg_depth = (uint16_t*)(&last_depth.data[0]);
            for(unsigned int ii = v-sz; ii <= v+sz; ++ii){
                for(unsigned int jj = u-sz; jj <= u+sz; ++jj){
                    uint16_t& T = reg_depth[ii*step + jj];
                    if(T > 700U && T < 3500U){
                        geometry_msgs::Point P;
                        P.z = T*0.001;
                        P.x = (jj-cx)*P.z/fx;
                        P.y = (ii-cy)*P.z/fy;
                        plane_3d_points.push_back(P);
                    }
                }
            }

            // std::cout << "Found : " << plane_3d_points.size() << " points" << std::endl;

            // Only accept point with more than 3 valid depths in its neighbors.....
            if(plane_3d_points.size() < sz){
                std::cerr << style::yellow << "     No Valid Depth Found" << style::normal << std::endl;
                return false;
            }

            return true;
        }


        bool updateIntersectionPoint2(const pupil_msgs::eye_statusConstPtr& eye_status_msg){
            debug("updateGazeDepthIntersectionPoints2()");

            if(eye_status_msg->gaze_normal_3d.z > -0.0001 && 0.0001 > eye_status_msg->gaze_normal_3d.z){
                // Gaze Normal in Z is Zero, Invalid Gaze. SKIP!!
                return false;
            }

            Eigen::Matrix<float,3,1> gazeline;
            gazeline << eye_status_msg->gaze_normal_3d.x,eye_status_msg->gaze_normal_3d.y,eye_status_msg->gaze_normal_3d.z;

            // Cleanup from previous iteration
            plane_3d_points.clear();
            unsigned int sz_ratio = 1;

            sensor_msgs::PointCloud2Iterator<float> iter_x(pointcloud, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(pointcloud, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(pointcloud, "z");
            sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(pointcloud, "r");
            sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(pointcloud, "g");
            sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(pointcloud, "b");
            sensor_msgs::PointCloud2Iterator<uint8_t> iter_a(pointcloud, "a");

            float gazeline_norm = gazeline.norm();
            Eigen::Matrix<float,3,1> eye_center;
            eye_center << eye_status_msg->eye_center_3d.x/1000.0,eye_status_msg->eye_center_3d.y/1000.0,eye_status_msg->eye_center_3d.z/1000.0;
            for(unsigned int uu=0; uu < pointcloud.height; ++uu){
                for(unsigned int vv=0; vv < pointcloud.width; ++vv, ++iter_x, ++iter_y, ++iter_z){
                    if(std::isnan(*iter_x)){
                        continue;
                    }
                    else{
                        Eigen::Matrix<float,3,1> point;
                        point << *iter_x,*iter_y,*iter_z;
                        point -= eye_center;
                        float distance = point.cross(gazeline).norm()/gazeline_norm;
                        if(distance < 0.1){
                            // std::cout << gazeline << '\n';
                            // std::cout << gazeline_norm << '\n';
                            // std::cout << point << '\n';
                            // std::cout << distance << "\n---------------\n";
                            geometry_msgs::Point p;
                            p.x = *iter_x;
                            p.y = *iter_y;
                            p.z = *iter_z;
                            plane_3d_points.push_back(p);
                        }
                    }
                }
            }
            // std::cout << std::endl;
            // unsigned int pc_size = pointcloud.row_step*pointcloud.height;
            // for(unsigned int ii=0; ii < pc_size; ++ii){
            //     Eigen::Matrix<float,3,1> point;
            //     point << pointcloud.data[ii].x-eye_status_msg->eye_center_3d.x,
            //              pointcloud.data[ii].y-eye_status_msg->eye_center_3d.y,
            //              pointcloud.data[ii].z-eye_status_msg->eye_center_3d.z;
            //     // CHANGED!
            //     // float distance = point.cross(gaze_line).norm()/point.norm();
            //     if(distance < 0.1){
            //         plane_3d_points.push_back(pointcloud.data[ii]);
            //     }
            // }


            // Only accept point with more than 3 valid depths in its neighbors.....
            if(plane_3d_points.size() < 20){
                std::cerr << style::yellow << "     No Valid Depth Found" << style::normal << std::endl;
                return false;
            }

            return true;
        }


//===========================================================================================================
//    #####                                                         ######                              ######
//   #     #   ##   #       ####  #    # #        ##   ##### ###### #     # #        ##   #    # ###### #     #  ####   ####  ######
//   #        #  #  #      #    # #    # #       #  #    #   #      #     # #       #  #  ##   # #      #     # #    # #      #
//   #       #    # #      #      #    # #      #    #   #   #####  ######  #      #    # # #  # #####  ######  #    #  ####  #####
//   #       ###### #      #      #    # #      ######   #   #      #       #      ###### #  # # #      #       #    #      # #
//   #     # #    # #      #    # #    # #      #    #   #   #      #       #      #    # #   ## #      #       #    # #    # #
//    #####  #    # ######  ####   ####  ###### #    #   #   ###### #       ###### #    # #    # ###### #        ####   ####  ######
//===========================================================================================================
        bool calculatePlanePose(geometry_msgs::Pose& new_plane_pose, double& confidence){
            debug("calculatePlane()");

            // Found valid depth values, calculate the MEAN distance from camera (depth) from stored plane_3d_points
            gaze_3d_msg.point.x = 0;
            gaze_3d_msg.point.y = 0;
            gaze_3d_msg.point.z = 0;
            for(const auto& it : plane_3d_points){
                gaze_3d_msg.point.x += it.x;
                gaze_3d_msg.point.y += it.y;
                gaze_3d_msg.point.z += it.z;
            }
            gaze_3d_msg.point.x /= (double)plane_3d_points.size();
            gaze_3d_msg.point.y /= (double)plane_3d_points.size();
            gaze_3d_msg.point.z /= (double)plane_3d_points.size();
            // 3D Point
            gaze_3d_pub.publish(gaze_3d_msg);

            // Populate the message with the marker for the intersection point for rviz
            point_marker_msg.pose.position.x = gaze_3d_msg.point.x;
            point_marker_msg.pose.position.y = gaze_3d_msg.point.y;
            point_marker_msg.pose.position.z = gaze_3d_msg.point.z;
            point_marker_pub.publish(point_marker_msg);

            // NOTE: gaze_3d_msg.point is effectively the mean point necessary for SVD based PCA
            Eigen::Matrix<double,3,3> C = Eigen::Matrix<double,3,3>::Zero(); // Covariance matrix
            for(auto& it : plane_3d_points){
                it.x -= gaze_3d_msg.point.x;
                it.y -= gaze_3d_msg.point.y;
                it.z -= gaze_3d_msg.point.z;
                // current iterator now has the vector between the current point and the mean point

                // Calculate the covariance matrix, element by element instead of the normal (x*x')
                // so there isnt a need to create a temporary data structure for the multiplication
                // NOTE: matrix is symetry, only constructed the top right part in the loop
                C.coeffRef(0,0) += it.x*it.x;
                C.coeffRef(1,1) += it.y*it.y;
                C.coeffRef(2,2) += it.z*it.z;
                C.coeffRef(0,1) += it.x*it.y;
                C.coeffRef(0,2) += it.x*it.z;
                C.coeffRef(1,2) += it.y*it.z;
            }
            // Fill the rest of the covariance matrix
            C.coeffRef(1,0) = C.coeffRef(0,1);
            C.coeffRef(2,0) = C.coeffRef(0,2);
            C.coeffRef(2,1) = C.coeffRef(1,2);

            // Calculate the Eigen vectors and eigen values, reorder vectors according to the diagonal values
            eig_solver.compute(C,true);
            Eigen::Matrix<std::complex<double>,3,1> val = eig_solver.eigenvalues();
            Eigen::Matrix<std::complex<double>,3,3> vec = eig_solver.eigenvectors();

            // TODO:
            // TODO: (A)
            // TODO:
            // std::chrono::high_resolution_clock::time_point local_t1, local_t2;
            // local_t1 = hr_clock::now();
            //
            // Eigen::Matrix<double,3,3> R;
            // std::array<int,3> idx_order;
            //
            // // Local define, Sorts idx_order according to the eigen values
            // #define order_check(a,b,c) val.coeffRef(a).real() >= val.coeffRef(b).real() && val.coeffRef(b).real() >= val.coeffRef(c).real()
            // if(order_check(0,1,2)) idx_order = {0,1,2};
            // else if(order_check(2,1,0)) idx_order = {2,1,0};
            // else if(order_check(1,0,2)) idx_order = {1,0,2};
            // else if(order_check(2,0,1)) idx_order = {2,0,1};
            // else if(order_check(1,2,0)) idx_order = {1,2,0};
            // else if(order_check(0,2,1)) idx_order = {0,2,1};
            // #undef order_check  // Not necessary after this
            //
            // // Sorted Eigen Values
            // double eig_vals[3] = {0,0,0};
            // for(int ii=0;ii<3;++ii){
            //     R.coeffRef(0,ii) = vec.coeffRef(0,idx_order[ii]).real();
            //     R.coeffRef(1,ii) = vec.coeffRef(1,idx_order[ii]).real();
            //     R.coeffRef(2,ii) = vec.coeffRef(2,idx_order[ii]).real();
            //     eig_vals[ii] = val.coeffRef(idx_order[ii]).real();
            // }
            //
            // // Calculate Confidence
            // double d0 = eig_vals[0]/eig_vals[2];
            // double d1 = eig_vals[0]/eig_vals[1];
            // confidence = ((d0-d1)/d0);
            // //std::cout << "("<<eig_vals[0] << ", " << eig_vals[1] << ", " << eig_vals[2] << ") C: " << confidence << std::endl;
            //
            // Eigen::Quaternion<double> q;
            // q.w() = sqrt(1 + R(0,0) + R(1,1) + R(2,2))/2.0;
            // q.x() = (R(2,1)-R(1,2))/(4*q.w());
            // q.y() = (R(0,2)-R(2,0))/(4*q.w());
            // q.z() = (R(1,0)-R(0,1))/(4*q.w());
            // if( !std::isnormal(q.w()) || !std::isnormal(q.x()) || !std::isnormal(q.y()) || !std::isnormal(q.z())){
            //     std::cerr << style::yellow << "     Plane intersection pose invalid." << style::normal << std::endl;
            //     return false;
            // }
            //
            //
            // // CHANGED: Z normal in oposition with gaze line
            // q.normalize();
            // Eigen::Matrix<double,3,1> off;
            // off << 0,0,1;
            // off = q._transformVector(off);
            //
            // if(off(2,0) < 0){
            //     // Necessary to transform the pose!
            //     Eigen::Quaternion<double> g_;
            //     g_ = Eigen::AngleAxis<double>(math::pi, Eigen::Matrix<double,3,1>::UnitX());
            //     q = q*g_;
            //     q.normalize();
            // }
            //
            // // CHANGED: X normal projection colinear with Camera X axis using Plane-Line Intersection
            // Eigen::Quaternion<double> horn_quat(horn_pose.orientation.w,horn_pose.orientation.x,horn_pose.orientation.y,horn_pose.orientation.z);
            // horn_quat.normalize();
            //
            // // Rotation Matrix instead of quaternions
            // Eigen::Matrix<double,3,3> rot_horn = horn_quat.toRotationMatrix();
            // Eigen::Matrix<double,3,3> rot_plane = q.toRotationMatrix();
            // // Z Axis Directions from both Horn and Plane Poses from rotation Matrix Columns
            // Eigen::Matrix<double,3,1> plane_normal = rot_plane.col(2);
            // Eigen::Matrix<double,3,1> ray_direction = rot_horn.col(2);
            //
            // // Line and Plane point Definitions
            // Eigen::Matrix<double,3,1> plane_point;
            // Eigen::Matrix<double,3,1> ray_point;
            // plane_point << gaze_3d_msg.point.x, gaze_3d_msg.point.y, gaze_3d_msg.point.z;
            // ray_point = plane_point + rot_horn.col(0);
            //
            // // Actual Line/Plane Intersection Algorithm
            // Eigen::Matrix<double,3,1> intersection = ray_point
            //     + ray_direction
            //         * ( -plane_normal.dot(ray_point-plane_point)
            //             /plane_normal.dot(ray_direction) );
            //
            // Eigen::Matrix<double,3,1> proj_point = q.inverse()._transformVector(intersection)-plane_point;
            // double rot_angle = std::atan2(proj_point[1],proj_point[0]);
            //
            // Eigen::Quaternion<double> q_corr;
            // q_corr = Eigen::AngleAxis<double>(rot_angle, Eigen::Matrix<double,3,1>::UnitZ());
            // q = q*q_corr;
            // q.normalize();


            //{ TODO: (A)
            // Remove lines from (A) to (A), verify if the following commented code results in the same

            // Extract the eigenvector column corresponding to the smallest eigenvalue
            unsigned int smallest_eigen_index = 0;
            double smallest_eigen_value = val.coeffRef(0).real();
            for(unsigned int ii=1; ii<3;++ii){
                double tmp_val = val.coeffRef(ii).real();
                if(tmp_val < smallest_eigen_value){
                    smallest_eigen_value = tmp_val;
                    smallest_eigen_index = ii;
                }
            }

            Eigen::Matrix<double,3,3> R;
            R.col(2) = vec.col(smallest_eigen_index).real();
            R.col(2).normalize();

            // Verify if the Gaze Z axis if oposed to the Camera Z axis, if it isnt rotate 180 degrees in X
            if( R.coeffRef(2,2) < 0 ){
                R.col(2) *= -1;
            }

            // Calculate the X axis versor with the R200 Horn camera Y versor and the Z gaze versor
            Eigen::Quaternion<double> horn_quat(horn_pose.orientation.w,horn_pose.orientation.x,horn_pose.orientation.y,horn_pose.orientation.z);
            Eigen::Matrix<double,3,3> rot_horn = horn_quat.toRotationMatrix();
            // NOTE: may be necessary to transpose some of the matrixes
            R.col(0) = -rot_horn.col(1).cross(R.col(2));

            // Calculate the Y gaze versor
            R.col(1) = R.col(2).cross(R.col(0));

            // NOTE: may be necessary to normalize the rotation matrix columns
            Eigen::Quaternion<double> q(R);

            // local_t2 = hr_clock::now();
            // std::cout << (double)std::chrono::duration_cast<std::chrono::nanoseconds>( local_t2 - local_t1 ).count() << std::endl;

            //} TODO:

            q.normalize();
            // Populate the passed-by-reference output
            new_plane_pose.position.x = gaze_3d_msg.point.x;
            new_plane_pose.position.y = gaze_3d_msg.point.y;
            new_plane_pose.position.z = gaze_3d_msg.point.z;
            new_plane_pose.orientation.w = q.w();
            new_plane_pose.orientation.x = q.x();
            new_plane_pose.orientation.y = q.y();
            new_plane_pose.orientation.z = q.z();

            return true;

        }

//================================1.0===========================================================================
//                                                                 #     #                      ######
//    ####    ##   #       ####  #    # #        ##   ##### ###### ##   ## ######   ##   #    # #     #  ####   ####  ######
//   #    #  #  #  #      #    # #    # #       #  #    #   #      # # # # #       #  #  ##   # #     # #    # #      #
//   #      #    # #      #      #    # #      #    #   #   #####  #  #  # #####  #    # # #  # ######  #    #  ####  #####
//   #      ###### #      #      #    # #      ######   #   #      #     # #      ###### #  # # #       #    #      # #
//   #    # #    # #      #    # #    # #      #    #   #   #      #     # #      #    # #   ## #       #    # #    # #
//    ####  #    # ######  ####   ####  ###### #    #   #   ###### #     # ###### #    # #    # #        ####   ####  ######
//===========================================================================================================
        void calculateMeanPose(geometry_msgs::Pose& mean_, CircularArray<GazeData,HISTORY_BUF_SIZE>& hist_){
            debug("calculateMeanPose()");

            double tot = 0.0;
            std::size_t fixation_buf_size = hist_.size();
            GazeData& cur_gaze = hist_[0];
            mean_.position.x = 0;
            mean_.position.y = 0;
            mean_.position.z = 0;
            mean_.orientation.w = 0;
            mean_.orientation.x = 0;
            mean_.orientation.y = 0;
            mean_.orientation.z = 0;
            for(std::size_t ii=1; ii < fixation_buf_size; ++ii){
                GazeData& gaze_elem = hist_[ii];
                if( gaze_elem.tm < cur_gaze.tm-MINIMUM_FIXATION_DURATION ){
                    break;
                }
                tot += gaze_elem.confidence;
                mean_.position.x += gaze_elem.pose.position.x*gaze_elem.confidence;
                mean_.position.y += gaze_elem.pose.position.y*gaze_elem.confidence;
                mean_.position.z += gaze_elem.pose.position.z*gaze_elem.confidence;
                mean_.orientation.w += gaze_elem.pose.orientation.w*gaze_elem.confidence;
                mean_.orientation.x += gaze_elem.pose.orientation.x*gaze_elem.confidence;
                mean_.orientation.y += gaze_elem.pose.orientation.y*gaze_elem.confidence;
                mean_.orientation.z += gaze_elem.pose.orientation.z*gaze_elem.confidence;
            }

            mean_.position.x /= tot;
            mean_.position.y /= tot;
            mean_.position.z /= tot;
            mean_.orientation.w /= tot;
            mean_.orientation.x /= tot;
            mean_.orientation.y /= tot;
            mean_.orientation.z /= tot;

            Eigen::Quaternion<double> q;
            q.vec() << mean_.orientation.x,mean_.orientation.y,mean_.orientation.z;
            q.w() = mean_.orientation.w;
            q.normalize();
            mean_.orientation.w = q.w();
            mean_.orientation.x = q.x();
            mean_.orientation.y = q.y();
            mean_.orientation.z = q.z();
        }

//===========================================================================================================
//                                                                 #     #               ######
//    ####    ##   #       ####  #    # #        ##   ##### ###### #     #   ##   #####  #     #  ####   ####  ######
//   #    #  #  #  #      #    # #    # #       #  #    #   #      #     #  #  #  #    # #     # #    # #      #
//   #      #    # #      #      #    # #      #    #   #   #####  #     # #    # #    # ######  #    #  ####  #####
//   #      ###### #      #      #    # #      ######   #   #       #   #  ###### #####  #       #    #      # #
//   #    # #    # #      #    # #    # #      #    #   #   #        # #   #    # #   #  #       #    # #    # #
//    ####  #    # ######  ####   ####  ###### #    #   #   ######    #    #    # #    # #        ####   ####  ######
//===========================================================================================================
        void calculateVarPose(geometry_msgs::Pose& var_, geometry_msgs::Pose& mean_, CircularArray<GazeData,HISTORY_BUF_SIZE>& hist_){
            debug("calculateVarPose()");

            double tot = 0.0;
            std::size_t fixation_buf_size = hist_.size();
            GazeData& cur_gaze = hist_[0];
            var_.position.x = 0;
            var_.position.y = 0;
            var_.position.z = 0;
            var_.orientation.w = 0;
            var_.orientation.x = 0;
            var_.orientation.y = 0;
            var_.orientation.z = 0;
            for(std::size_t ii=0; ii < fixation_buf_size; ++ii){
                GazeData& gaze_elem = hist_[ii];
                if( gaze_elem.tm < cur_gaze.tm-MINIMUM_FIXATION_DURATION ){
                    break;
                }
                tot += gaze_elem.confidence;
                var_.position.x += (gaze_elem.pose.position.x - mean_.position.x)*(gaze_elem.pose.position.x - mean_.position.x)*gaze_elem.confidence;
                var_.position.y += (gaze_elem.pose.position.y - mean_.position.y)*(gaze_elem.pose.position.y - mean_.position.y)*gaze_elem.confidence;
                var_.position.z += (gaze_elem.pose.position.z - mean_.position.z)*(gaze_elem.pose.position.z - mean_.position.z)*gaze_elem.confidence;
                var_.orientation.w += (gaze_elem.pose.orientation.w - mean_.orientation.w)*(gaze_elem.pose.orientation.w - mean_.orientation.w)*gaze_elem.confidence;
                var_.orientation.x += (gaze_elem.pose.orientation.x - mean_.orientation.x)*(gaze_elem.pose.orientation.x - mean_.orientation.x)*gaze_elem.confidence;
                var_.orientation.y += (gaze_elem.pose.orientation.y - mean_.orientation.y)*(gaze_elem.pose.orientation.y - mean_.orientation.y)*gaze_elem.confidence;
                var_.orientation.z += (gaze_elem.pose.orientation.z - mean_.orientation.z)*(gaze_elem.pose.orientation.z - mean_.orientation.z)*gaze_elem.confidence;
            }
            var_.position.x = sqrt(var_.position.x/tot);
            var_.position.y = sqrt(var_.position.y/tot);
            var_.position.z = sqrt(var_.position.z/tot);
            var_.orientation.w = sqrt(var_.orientation.w/tot);
            var_.orientation.x = sqrt(var_.orientation.x/tot);
            var_.orientation.y = sqrt(var_.orientation.y/tot);
            var_.orientation.z = sqrt(var_.orientation.z/tot);
        }

//===========================================================================================================
//      #                                                                  ######
//     # #    ####   ####  ###### #####  #####   ##   #####  #      ###### #     # #  ####  #####  ###### #####   ####  #  ####  #    #
//    #   #  #    # #    # #      #    #   #    #  #  #    # #      #      #     # # #      #    # #      #    # #      # #    # ##   #
//   #     # #      #      #####  #    #   #   #    # #####  #      #####  #     # #  ####  #    # #####  #    #  ####  # #    # # #  #
//   ####### #      #      #      #####    #   ###### #    # #      #      #     # #      # #####  #      #####       # # #    # #  # #
//   #     # #    # #    # #      #        #   #    # #    # #      #      #     # # #    # #      #      #   #  #    # # #    # #   ##
//   #     #  ####   ####  ###### #        #   #    # #####  ###### ###### ######  #  ####  #      ###### #    #  ####  #  ####  #    #
//===========================================================================================================
        bool acceptableVariance(geometry_msgs::Pose& var_,bool only_tr = true){
            debug("acceptableVariance()");
            printPose(var_);
            if(only_tr){
                if( normPositionVariance(var_) < GAZE_POSITION_THRESH ){
                    std::cout << style::green << "   Acceptable variance" << style::normal << std::endl;
                    return true;
                }
            }
            else{
                if( var_.position.x < GAZE_POSITION_THRESH
                        && var_.position.y < GAZE_POSITION_THRESH
                        && var_.position.z < GAZE_POSITION_THRESH
                        && var_.orientation.x < GAZE_ORIENTATION_THRESH
                        && var_.orientation.y < GAZE_ORIENTATION_THRESH
                        && var_.orientation.z < GAZE_ORIENTATION_THRESH ){
                    std::cout << style::green << "   Acceptable variance" << style::normal << std::endl;
                    return true;
                }
            }

            std::cout << style::yellow << "   Non-Acceptable variance" << style::normal << std::endl;
            return false;
        }

        double normPositionVariance(geometry_msgs::Pose& var_){
            return sqrt(var_.position.x*var_.position.x + var_.position.y*var_.position.y + var_.position.z*var_.position.z);
        }

//===========================================================================================================
//   #######                              ######
//   #       # #      ##### ###### #####  #     #  ####   ####  ######
//   #       # #        #   #      #    # #     # #    # #      #
//   #####   # #        #   #####  #    # ######  #    #  ####  #####
//   #       # #        #   #      #####  #       #    #      # #
//   #       # #        #   #      #   #  #       #    # #    # #
//   #       # ######   #   ###### #    # #        ####   ####  ######
//===========================================================================================================
        void filterPose(geometry_msgs::Pose& pose, std::vector<filter::LowPass<double>>& tr_, std::vector<filter::LowPass<double>>& rot_, bool set_=false){

            Eigen::Quaternion<double> q_tmp;
            q_tmp.vec() << rot_[0].filter(pose.orientation.x), rot_[1].filter(pose.orientation.y), rot_[2].filter(pose.orientation.z);
            q_tmp.w() = rot_[3].filter(pose.orientation.w);
            geometry_msgs::PoseStamped stamped_pose;
            stamped_pose.header.stamp = last_tf_time;
            stamped_pose.header.frame_id = "rgb_optical_frame";

            q_tmp.normalize();

            if(set_){
                tr_[0].set(pose.position.x);
                tr_[1].set(pose.position.y);
                tr_[2].set(pose.position.z);
                rot_[0].set(q_tmp.x());
                rot_[1].set(q_tmp.y());
                rot_[2].set(q_tmp.z());
                rot_[3].set(q_tmp.w());
            }
            else{
                pose.position.x = tr_[0].filter(pose.position.x);
                pose.position.y = tr_[1].filter(pose.position.y);
                pose.position.z = tr_[2].filter(pose.position.z);
                pose.orientation.x = q_tmp.x();
                pose.orientation.y = q_tmp.y();
                pose.orientation.z = q_tmp.z();
                pose.orientation.w = q_tmp.w();
            }
        }

//===========================================================================================================
//   ######                                        #####
//   #     # #    # #####  #      #  ####  #    # #     #  ####    ##   #
//   #     # #    # #    # #      # #      #    # #       #    #  #  #  #
//   ######  #    # #####  #      #  ####  ###### #  #### #    # #    # #
//   #       #    # #    # #      #      # #    # #     # #    # ###### #
//   #       #    # #    # #      # #    # #    # #     # #    # #    # #
//   #        ####  #####  ###### #  ####  #    #  #####   ####  #    # ######
//===========================================================================================================
        void publishGoal(geometry_msgs::Pose& mean_pose,double height=0.15){

            // Ugly transformations
            Eigen::Matrix<double,3,1> vec;
            vec << 0,0,-height;
            Eigen::Quaternion<double> quat(mean_pose.orientation.w,mean_pose.orientation.x,mean_pose.orientation.y,mean_pose.orientation.z);
            vec = quat._transformVector(vec);
            mean_pose.position.x += vec(0);
            mean_pose.position.y += vec(1);
            mean_pose.position.z += vec(2);

            goal_pose.header.seq++;
            goal_pose.header.stamp = gaze_3d_msg.header.stamp;
            goal_pose.header.frame_id = "jaco";
            goal_pose.pose = mean_pose;
			goal_pose_pub.publish(goal_pose);

            kst_goal.add(goal_pose.header.stamp.toSec()-start_time, 1.0, goal_pose.pose);
            beep.sound.play();

            // listener.transformPose("jaco", stamped_pose, goal_pose);
            std::cout << style::magenta << "   Goal: " << style::normal;
			printPose(goal_pose.pose); std::cout << std::endl;

	        double qw_ = mean_pose.orientation.w;
			double qz_ = mean_pose.orientation.z;
			double qy_ = mean_pose.orientation.y;
			double qx_ = mean_pose.orientation.x;
			double tx_ = atan2((2 * qw_ * qx_ - 2 * qy_ * qz_), (qw_ * qw_ - qx_ * qx_ - qy_ * qy_ + qz_ * qz_));
			double ty_ = asin(2 * qw_ * qy_ + 2 * qx_ * qz_);
			double tz_ = atan2((2 * qw_ * qz_ - 2 * qx_ * qy_), (qw_ * qw_ + qx_ * qx_ - qy_ * qy_ - qz_ * qz_));

            kinova_msgs::GoToPose srv_msg;
            srv_msg.request.X = goal_pose.pose.position.x;
            srv_msg.request.Y = goal_pose.pose.position.y;
            srv_msg.request.Z = goal_pose.pose.position.z;
            srv_msg.request.ThetaX = tx_;
            srv_msg.request.ThetaY = ty_;
            srv_msg.request.ThetaZ = tz_;
            goal_srv.call(srv_msg);
        }

    };

    PLUGINLIB_DECLARE_CLASS(pupil_utils, GazeNodelet, pupil_utils::GazeNodelet, nodelet::Nodelet);

}
