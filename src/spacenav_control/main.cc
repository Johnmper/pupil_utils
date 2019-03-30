// STANDARD C++ INCLUDES
#include<iostream>
#include<string>
#include<sstream>
#include<cmath>
// ROS INCLUDES
#include<ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include<sensor_msgs/Joy.h>
#include<gazebo_msgs/ModelState.h>
#include<gazebo_msgs/ModelStates.h>
#include<tf/transform_broadcaster.h>

#include<Eigen/Eigen>

#include<ros_utils/filter.hh>
#include<ros_utils/style.hh>
#include<ros_utils/math.hh>

#define REALSENSE_MODEL_NAME "realsense_camera"
#define POSE_TR_MEAN   0.0
#define POSE_TR_VAR    0.0
#define POSE_ROT_MEAN  0.0
#define POSE_ROT_VAR   0.0

#define KALMAN_UPDATE_FREQ      2
#define KALMAN_PREDICTION_FREQ  4

//===========================================================================================================
//    #####                                                    #####
//   #     # #####    ##    ####  ###### #    #   ##   #    # #     #  ####  #    # ##### #####   ####  #
//   #       #    #  #  #  #    # #      ##   #  #  #  #    # #       #    # ##   #   #   #    # #    # #
//    #####  #    # #    # #      #####  # #  # #    # #    # #       #    # # #  #   #   #    # #    # #
//         # #####  ###### #      #      #  # # ###### #    # #       #    # #  # #   #   #####  #    # #
//   #     # #      #    # #    # #      #   ## #    #  #  #  #     # #    # #   ##   #   #   #  #    # #
//    #####  #      #    #  ####  ###### #    # #    #   ##    #####   ####  #    #   #   #    #  ####  ######
//===========================================================================================================
class SpacenavControl{
public:

    /// Realsense camera trutch position Marker Publisher
    ros::Publisher rs_truth_pub;
    /// Realsense camera truth  position Marker msg
    visualization_msgs::Marker rs_truth_msg;
    /// Realsense camera last measurment position Marker Publisher
    ros::Publisher rs_last_z_pub;
    /// Realsense camera last measurment position Marker msg
    visualization_msgs::Marker rs_last_z_msg;


    /// Kalman filter for translation
    filter::Kalman<float,9,3> translation_kalman;
    /// Kalman filter for rotation
    filter::Kalman<float,12,4> rotation_kalman;

    /// Rotation artificial noisy measurement
    Eigen::Matrix<float,4,1> rotation_noisy;
    /// Translation artificial noisy measurement
    Eigen::Matrix<float,3,1> translation_noisy;

    /// Handle of ros node, only needed for class contruction
    ros::NodeHandle     node;
    /// Subscriber for the space mouse topic
    ros::Subscriber     space_mouse_sub;
    /// Subscriber to read current realsense camera pose in simulation
    ros::Subscriber     realsense_modelstates_sub;
    /// Publisher for control of the realsense camera pose in gazebo
    ros::Publisher      realsense_modelstate_pub;

    /// Current Space mouse mean position
    std::vector<float> cur;
    /// Space mouse gains
    std::vector<float> gains;
    /// Current pose of RS Camera
    gazebo_msgs::ModelState rs_state;

    /// Current translation in simulation relation to the jaco frame
    Eigen::Matrix<float,3,1> translation;
    /// Current rotation in simulation relation to the jaco frame
    Eigen::Quaternion<float> rotation;
    /// Current space mouse tranlsation
    Eigen::Matrix<float,3,1> mouse_translation;
    /// Current space mouse rotation quaternion
    Eigen::Quaternion<float> mouse_rotation;


    /// [jaco -> camera_link] transform publisher
    tf::TransformBroadcaster tf_broadcaster;
    /// Smaller piece of the Transform message
    geometry_msgs::Transform tf_camera_transform;
    /// Actual [jaco -> camera_link] transform message
    geometry_msgs::TransformStamped tf_camera_link;

    /// ROS Timer for publishing the camera transform
    ros::Timer kalman_timer;

    /// Necessary object for the random generators
    std::random_device rd;
    /// Necessary object for the random generators
    std::mt19937 gen;
    /// Normal distribution generator
    std::normal_distribution<double> d_tr;
    /// Normal distribution generator
    std::normal_distribution<double> d_rot;
//===========================================================================================================
//    #####
//   #     #  ####  #    #  ####  ##### #####  #    #  ####  #####  ####  #####
//   #       #    # ##   # #        #   #    # #    # #    #   #   #    # #    #
//   #       #    # # #  #  ####    #   #    # #    # #        #   #    # #    #
//   #       #    # #  # #      #   #   #####  #    # #        #   #    # #####
//   #     # #    # #   ## #    #   #   #   #  #    # #    #   #   #    # #   #
//    #####   ####  #    #  ####    #   #    #  ####   ####    #    ####  #    #
//===========================================================================================================
    SpacenavControl() : rd(), gen(rd()), d_tr(POSE_TR_MEAN,POSE_TR_VAR), d_rot(POSE_ROT_MEAN,POSE_ROT_VAR){

        // Constant values of new state
        rs_state.model_name = REALSENSE_MODEL_NAME;
        rs_state.pose.position.x = -0.882355868816;
        rs_state.pose.position.y = -1.38669633865;
        rs_state.pose.position.z = 1.34402990341;
        rs_state.pose.orientation.x = -0.0824097535059;
        rs_state.pose.orientation.y = 0.0761110103855;
        rs_state.pose.orientation.z = 0.343800600632;
        rs_state.pose.orientation.w = 0.932318021723;

        // Initialization of Camera TF
        tf_camera_link.header.frame_id = "world";
        tf_camera_link.child_frame_id = "r200_camera_link";
        // Update the Realsense camera TF
        tf_camera_link.transform.translation.x = rs_state.pose.position.x;
        tf_camera_link.transform.translation.y = rs_state.pose.position.y;
        tf_camera_link.transform.translation.z = rs_state.pose.position.z;
        tf_camera_link.transform.rotation = rs_state.pose.orientation;
        tf_camera_link.header.stamp = ros::Time::now();

        // Initiate the markers messages for Rviz
        rs_truth_msg.header.seq = 0;
        rs_truth_msg.header.frame_id = "world";
        rs_truth_msg.ns = "gaze_markers";
        rs_truth_msg.id = 0;
        rs_truth_msg.type = visualization_msgs::Marker::SPHERE;
        rs_truth_msg.action = visualization_msgs::Marker::ADD;
        rs_truth_msg.pose.orientation.x = 0;
        rs_truth_msg.pose.orientation.y = 0;
        rs_truth_msg.pose.orientation.z = 0;
        rs_truth_msg.pose.orientation.w = 1;
        rs_truth_msg.pose.position.x = rs_state.pose.position.x;
        rs_truth_msg.pose.position.y = rs_state.pose.position.y;
        rs_truth_msg.pose.position.z = rs_state.pose.position.z;
        rs_truth_msg.scale.x = 0.04;
        rs_truth_msg.scale.y = 0.04;
        rs_truth_msg.scale.z = 0.04;
        rs_truth_msg.color.r = 0;
        rs_truth_msg.color.g = 1;
        rs_truth_msg.color.b = 0;
        rs_truth_msg.color.a = 1;

        rs_last_z_msg = rs_truth_msg;
        rs_last_z_msg.id = 1;
        rs_last_z_msg.color.r = 1;
        rs_last_z_msg.color.b = 0;

        // Initialization of the camera link translation and rotation
        translation.coeffRef(0) = rs_state.pose.position.x;
        translation.coeffRef(1) = rs_state.pose.position.y;
        translation.coeffRef(2) = rs_state.pose.position.z;
        rotation.x() = rs_state.pose.orientation.x;
        rotation.y() = rs_state.pose.orientation.y;
        rotation.z() = rs_state.pose.orientation.z;
        rotation.w() = rs_state.pose.orientation.w;

        // Initialization of the mouse displacemente pose
        mouse_translation << 0,0,0;
        mouse_rotation.vec() << 0,0,0;
        mouse_rotation.w() = 1;

        // Pre allocation to size 6
        cur.resize(6,0);
        gains.resize(6,0);
        gains[0] = 0.0025;
        gains[1] = 0.0025;
        gains[2] = 0.0025;
        gains[3] = 0.0025;
        gains[4] = 0.0025;
        gains[5] = 0.0025;


        // Set publisher for control of realsense pose in gazebo simulation
        realsense_modelstate_pub = node.advertise<gazebo_msgs::ModelState>("gazebo/set_model_state",1);
        //
        rs_truth_pub = node.advertise<visualization_msgs::Marker>("pupil/gaze_0/camera_marker",0);
        rs_last_z_pub = node.advertise<visualization_msgs::Marker>("pupil/gaze_0/last_z_marker",0);

        const float prediction_frequency = 50.0;
        // Linear movement transition state
        translation_kalman.A.topRightCorner(6,6) += (1/prediction_frequency)*Eigen::Matrix<float,6,6>::Identity();
        translation_kalman.A.topRightCorner(3,3) += 0.5*(1/prediction_frequency)*(1/prediction_frequency)*Eigen::Matrix<float,3,3>::Identity();
        translation_kalman.Q = 0.01*Eigen::Matrix<float,9,9>::Identity();
        translation_kalman.R = 0.5*Eigen::Matrix<float,3,3>::Identity();
        translation_kalman.x << rs_state.pose.position.x,rs_state.pose.position.y,rs_state.pose.position.z,0,0,0,0,0,0;
        translation_kalman.showInfo();

        // Linear movement transition state
        rotation_kalman.A.topRightCorner(8,8) += (1/prediction_frequency)*Eigen::Matrix<float,8,8>::Identity();
        rotation_kalman.A.topRightCorner(4,4) += 0.5*(1/prediction_frequency)*(1/prediction_frequency)*Eigen::Matrix<float,4,4>::Identity();
        rotation_kalman.Q = 0.01*Eigen::Matrix<float,12,12>::Identity();
        rotation_kalman.R = 0.02*Eigen::Matrix<float,4,4>::Identity();
        rotation_kalman.x << rs_state.pose.orientation.x,rs_state.pose.orientation.y,rs_state.pose.orientation.z,rs_state.pose.orientation.w,0,0,0,0,0,0,0,0;
        rotation_kalman.showInfo();

        kalman_timer = node.createTimer(ros::Duration(1/prediction_frequency), &SpacenavControl::kalmanCallback,this); // 50Hz

        // Subscribe to the space mouse joy topic
        space_mouse_sub = node.subscribe("spacenav/joy",1,&SpacenavControl::spaceMouseCallback,this);

        // Subscribe to the gazebo model_states topic, it gives the state of every model simulated
        realsense_modelstates_sub = node.subscribe("gazebo/model_states",1, &SpacenavControl::modelStatesCallback,this);


    }
    ~SpacenavControl(){}


//===========================================================================================================
//   #    #                                     #####
//   #   #    ##   #      #    #   ##   #    # #     #   ##   #      #      #####    ##    ####  #    #
//   #  #    #  #  #      ##  ##  #  #  ##   # #        #  #  #      #      #    #  #  #  #    # #   #
//   ###    #    # #      # ## # #    # # #  # #       #    # #      #      #####  #    # #      ####
//   #  #   ###### #      #    # ###### #  # # #       ###### #      #      #    # ###### #      #  #
//   #   #  #    # #      #    # #    # #   ## #     # #    # #      #      #    # #    # #    # #   #
//   #    # #    # ###### #    # #    # #    #  #####  #    # ###### ###### #####  #    #  ####  #    #
//===========================================================================================================
private:
    void kalmanCallback(const ros::TimerEvent& event){
        static int counter = 0;

        Eigen::Quaternion<float> updated_rotation;
        Eigen::Matrix<float,3,1> updated_translation;

        // Update Kalman filter
        if( counter < (KALMAN_PREDICTION_FREQ/KALMAN_UPDATE_FREQ) ){
            // Only Prediction stage of the kalman filter
            rotation_kalman.predict();
            translation_kalman.predict();
        }

        if(counter >= (KALMAN_PREDICTION_FREQ/KALMAN_UPDATE_FREQ)){
            // Add noise to current simulation rotation
            rotation_noisy.coeffRef(0) = rotation.x() + d_rot(gen);
            rotation_noisy.coeffRef(1) = rotation.y() + d_rot(gen);
            rotation_noisy.coeffRef(2) = rotation.z() + d_rot(gen);
            rotation_noisy.coeffRef(3) = rotation.w() + d_rot(gen);
            // Add noise to current simulationn translation
            translation_noisy.coeffRef(0) = d_tr(gen) + translation.coeffRef(0);
            translation_noisy.coeffRef(1) = d_tr(gen) + translation.coeffRef(1);
            translation_noisy.coeffRef(2) = d_tr(gen) + translation.coeffRef(2);
            // Update kalman Rotation/Translation filter
            rotation_kalman.update(rotation_noisy);
            translation_kalman.update(translation_noisy);

            counter = 0;
        }

        // Extract translation and rotation from the kalman filtere internal state
        Eigen::Matrix<float,4,1> update_rotation_mat = rotation_kalman.state();
        updated_rotation.vec() << update_rotation_mat.coeffRef(0),update_rotation_mat.coeffRef(1),update_rotation_mat.coeffRef(2);
        updated_rotation.w() = update_rotation_mat.coeffRef(3);
        updated_rotation.normalize();
        updated_translation = translation_kalman.state();

        // Update the Realsense camera TF
        tf_camera_link.transform.translation.x = updated_translation.coeffRef(0);
        tf_camera_link.transform.translation.y = updated_translation.coeffRef(1);
        tf_camera_link.transform.translation.z = updated_translation.coeffRef(2);
        tf_camera_link.transform.rotation.x = updated_rotation.x();
        tf_camera_link.transform.rotation.y = updated_rotation.y();
        tf_camera_link.transform.rotation.z = updated_rotation.z();
        tf_camera_link.transform.rotation.w = updated_rotation.w();
        tf_camera_link.header.stamp = ros::Time::now();
        tf_camera_link.header.seq++;

        tf_broadcaster.sendTransform(tf_camera_link);

        rs_truth_msg.pose.position.x = translation.coeffRef(0);
        rs_truth_msg.pose.position.y = translation.coeffRef(1);
        rs_truth_msg.pose.position.z = translation.coeffRef(2);
        rs_truth_pub.publish(rs_truth_msg);

        rs_last_z_msg.pose.position.x = translation_noisy.coeffRef(0);
        rs_last_z_msg.pose.position.y = translation_noisy.coeffRef(1);
        rs_last_z_msg.pose.position.z = translation_noisy.coeffRef(2);
        rs_last_z_pub.publish(rs_last_z_msg);

        counter++;
    }

//===========================================================================================================
//    #####                              #     #                              #####
//   #     # #####    ##    ####  ###### ##   ##  ####  #    #  ####  ###### #     #   ##   #      #      #####    ##    ####  #    #
//   #       #    #  #  #  #    # #      # # # # #    # #    # #      #      #        #  #  #      #      #    #  #  #  #    # #   #
//    #####  #    # #    # #      #####  #  #  # #    # #    #  ####  #####  #       #    # #      #      #####  #    # #      ####
//         # #####  ###### #      #      #     # #    # #    #      # #      #       ###### #      #      #    # ###### #      #  #
//   #     # #      #    # #    # #      #     # #    # #    # #    # #      #     # #    # #      #      #    # #    # #    # #   #
//    #####  #      #    #  ####  ###### #     #  ####   ####   ####  ######  #####  #    # ###### ###### #####  #    #  ####  #    #
//===========================================================================================================
    void spaceMouseCallback(const sensor_msgs::Joy::ConstPtr& msg){
        static const float thresh = 0.1;
        //NOTE: Testing the mouse the max value obtained are about 0.68 so lets threshold at 0.1 and divide by 0.7 to
        // end up with a [0,1] value

        for(int ii=0; ii < 6; ++ii){
            if(msg->axes[ii] > thresh ){
                // Position values
                cur[ii] = gains[ii]*(msg->axes[ii]-thresh)/(0.686-thresh);
            }
            else if(msg->axes[ii] < -thresh ){
                // Negative values
                cur[ii] = gains[ii]*(msg->axes[ii]+thresh)/(0.686-thresh);
            }
            else{
                cur[ii] = 0;
            }
        }

        // Update mouse displacement pose
        mouse_translation << cur[0], cur[1], cur[2];
        mouse_rotation = Eigen::Quaternion<float>(math::eulerXYZ<float>(cur[3],cur[4],cur[5]));

        if(mouse_rotation.w() < 0){
            mouse_rotation.vec() *= -1;
            mouse_rotation.w() *= -1;
        }
        mouse_rotation.normalize();
    }

//===========================================================================================================
//   #     #                              #####                                    #####
//   ##   ##  ####  #####  ###### #      #     # #####   ##   ##### ######  ####  #     #   ##   #      #      #####    ##    ####  #    #
//   # # # # #    # #    # #      #      #         #    #  #    #   #      #      #        #  #  #      #      #    #  #  #  #    # #   #
//   #  #  # #    # #    # #####  #       #####    #   #    #   #   #####   ####  #       #    # #      #      #####  #    # #      ####
//   #     # #    # #    # #      #            #   #   ######   #   #           # #       ###### #      #      #    # ###### #      #  #
//   #     # #    # #    # #      #      #     #   #   #    #   #   #      #    # #     # #    # #      #      #    # #    # #    # #   #
//   #     #  ####  #####  ###### ######  #####    #   #    #   #   ######  ####   #####  #    # ###### ###### #####  #    #  ####  #    #
//===========================================================================================================

    void modelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr& msg){

        bool rs_found = false;
        for(int ii=0; ii < (msg->name).size(); ++ii){

            // Find realsense camera model searching by its name
            if((msg->name[ii]).compare(rs_state.model_name) == 0){
                rs_found = true;

                translation << msg->pose[ii].position.x, msg->pose[ii].position.y, msg->pose[ii].position.z;
                rotation.vec() << msg->pose[ii].orientation.x,msg->pose[ii].orientation.y,msg->pose[ii].orientation.z;
                rotation.w() = msg->pose[ii].orientation.w;
                if(rotation.w() < 0 ){
                    rotation.vec() *= -1;
                    rotation.w() *= -1;
                }

                rotation = rotation*mouse_rotation.inverse();
                if(rotation.w() < 0 ){
                    rotation.vec() *= -1;
                    rotation.w() *= -1;
                }
                rotation.normalize();

                translation = translation + rotation._transformVector(mouse_translation);

                // Populate the new position for the realsense camera in the simulation
                rs_state.pose.position.x    = translation.coeffRef(0);
                rs_state.pose.position.y    = translation.coeffRef(1);
                rs_state.pose.position.z    = translation.coeffRef(2);
                rs_state.pose.orientation.x = rotation.x();
                rs_state.pose.orientation.y = rotation.y();
                rs_state.pose.orientation.z = rotation.z();
                rs_state.pose.orientation.w = rotation.w();

                realsense_modelstate_pub.publish(rs_state);
            }
        }

        // Realsense camera was not found in the simulation, check if the model name defined  is correct
        if(!rs_found){
            ROS_ERROR_STREAM("Realsense camera not found in the simulation, searched for ["<< rs_state.model_name <<"]");
        }
    }
};


//===========================================================================================================
//
//   #    #   ##   # #    #
//   ##  ##  #  #  # ##   #
//   # ## # #    # # # #  #
//   #    # ###### # #  # #
//   #    # #    # # #   ##
//   #    # #    # # #    #
//===========================================================================================================
int main(int argc, char** argv){
    ROS_INFO_STREAM(LOG_ID("spacenav_control")<<"Starting node");

    ros::init(argc,argv,"spacenav_control");

    // Control of the realsense camera in the simulation with the spacenav mouse
    SpacenavControl mouse;
    ros::spin();

    ROS_INFO_STREAM(LOG_ID("spacenav_control")<<"Ending node");
    return 0;
}
