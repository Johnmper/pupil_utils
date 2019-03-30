#include <iostream>
#include <fstream>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <kinova_msgs/GoToPose.h>
#include <kinova_driver/kinova_comm.h>

#include <ros_utils/math.hh>

#include <Eigen/Eigen>

struct KstJaco{
    std::string filepath;
    std::ofstream file;

    KstJaco(std::string filepath_) : filepath(filepath_){
        file.open(filepath,std::ofstream::out);
        if(file.fail()){
            std::cerr << "Failed to create " << filepath << " file." << std::endl;
        }else{
            file << "time, tx, ty, tz, qx, qy, qz, qw";
            file << std::endl;
        }
    }
    ~KstJaco(){
        file.flush();
        file.close();
    }

    bool add(double tm, double x, double y, double z, double qx, double qy, double qz, double qw){

        if(!file.is_open()){
            std::cerr << "File isn't opened or was prematurely closed" << std::endl;
            return false;
        }

        file << tm << ", ";
        file << x << ", ";
        file << y << ", ";
        file << z << ", ";
        file << qx << ", ";
        file << qy << ", ";
        file << qz << ", ";
        file << qw << '\n';


        return true;
    }
};


struct KstJacoGoal{
    std::string filepath;
    std::ofstream file;

    KstJacoGoal(std::string filepath_) : filepath(filepath_){
        file.open(filepath,std::ofstream::out);
        if(file.fail()){
            std::cerr << "Failed to create " << filepath << " file." << std::endl;
        }else{
            file << "time, x_d, y_d, z_d, qx_d, qy_d, qz_d, qw_d";
            file << std::endl;
        }
    }
    ~KstJacoGoal(){
        file.flush();
        file.close();
    }

    bool add(double tm, double x, double y, double z, double qx, double qy, double qz, double qw){

        if(!file.is_open()){
            std::cerr << "File isn't opened or was prematurely closed" << std::endl;
            return false;
        }

        file << tm << ", ";
        file << x << ", ";
        file << y << ", ";
        file << z << ", ";
        file << qx << ", ";
        file << qy << ", ";
        file << qz << ", ";
        file << qw << '\n';

        return true;
    }
};

class SendPose{
public:
    KstJaco         kst_;
    KstJacoGoal     kst_goal_;
    double          start_time_;
    double          frequency_;
    double          cur_time_;
    ros::Timer      rt_timer_;
    ros::Timer      send_pose_timer_;
    ros::Publisher  joints_pub_;
    ros::ServiceServer srv_;
    sensor_msgs::JointState joints_msg_;

    boost::recursive_mutex api_mutex;
    std::unique_ptr<kinova::KinovaComm> comm_;

    kinova::KinovaPose pose;

    double last_goal_time = 0;

    SendPose(ros::NodeHandle& nh_) : frequency_(50.0), kst_goal_("/home/johnmper/.ROSData/jaco_control/kinova_goal.csv"), kst_("/home/johnmper/.ROSData/jaco_control/kinova.csv"){

        srv_ = nh_.advertiseService("/go_to_pose",&SendPose::recvPoseGoal,this);

        comm_.reset(new kinova::KinovaComm(nh_, api_mutex, true, "j2n6s300"));
        pose.X = 0.2124;
        pose.Y = -0.2579;
        pose.Z = 0.5085;
        pose.ThetaX = 1.64305;
        pose.ThetaY = 1.11679;
        pose.ThetaZ = 0.129651;

        joints_msg_.name.resize(9);
        joints_msg_.position.resize(9);
        joints_msg_.velocity.resize(9);
        joints_msg_.effort.resize(9);

        // Fill the Joint names
        for (int i = 0; i < 6; i++){
            joints_msg_.name[i] = "j2n6s300_joint_" + std::to_string(i+1);
        }
        for (int i = 0; i < 3; i++){
            joints_msg_.name[6+i] = "j2n6s300_joint_finger_" + std::to_string(i+1);
        }

        // Publisher for RVIZ joints
        joints_pub_ = nh_.advertise<sensor_msgs::JointState>("/j2n6s300/joint_states", 1);
        start_time_ = ros::Time::now().toSec();
        comm_->SetTorqueControlState(0);
        rt_timer_ = nh_.createTimer(ros::Duration(1/frequency_), &SendPose::updateRobot,this);
        comm_->homeArm();

        Eigen::Quaternion<double> g_;
        g_ = Eigen::AngleAxis<double>(pose.ThetaX, Eigen::Matrix<double,3,1>::UnitX())
            * Eigen::AngleAxis<double>(pose.ThetaY, Eigen::Matrix<double,3,1>::UnitY())
            * Eigen::AngleAxis<double>(pose.ThetaZ, Eigen::Matrix<double,3,1>::UnitZ());
        if(g_.w() < 0){
            g_.w() *= -1;
            g_.vec() *= -1;
        }
        kst_goal_.add(0,pose.X,pose.Y,pose.Z,g_.x(),g_.y(),g_.z(),g_.w());

    }

    ~SendPose(){
        Eigen::Quaternion<double> g_;
        g_ = Eigen::AngleAxis<double>(pose.ThetaX, Eigen::Matrix<double,3,1>::UnitX())
            * Eigen::AngleAxis<double>(pose.ThetaY, Eigen::Matrix<double,3,1>::UnitY())
            * Eigen::AngleAxis<double>(pose.ThetaZ, Eigen::Matrix<double,3,1>::UnitZ());
        if(g_.w() < 0){
            g_.w() *= -1;
            g_.vec() *= -1;
        }
        kst_goal_.add(cur_time_-start_time_,pose.X,pose.Y,pose.Z,g_.x(),g_.y(),g_.z(),g_.w());
    }

    bool recvPoseGoal(kinova_msgs::GoToPose::Request &req, kinova_msgs::GoToPose::Response &res)
    {
        last_goal_time = ros::Time::now().toSec();

        Eigen::Quaternion<double> g_;
        g_ = Eigen::AngleAxis<double>(pose.ThetaX, Eigen::Matrix<double,3,1>::UnitX())
            * Eigen::AngleAxis<double>(pose.ThetaY, Eigen::Matrix<double,3,1>::UnitY())
            * Eigen::AngleAxis<double>(pose.ThetaZ, Eigen::Matrix<double,3,1>::UnitZ());
        if(g_.w() < 0){
            g_.w() *= -1;
            g_.vec() *= -1;
        }
        kst_goal_.add(last_goal_time-start_time_-0.001,pose.X,pose.Y,pose.Z,g_.x(),g_.y(),g_.z(),g_.w());

        pose.X = req.X;
        pose.Y = req.Y;
        pose.Z = req.Z-0.05;
        if(pose.Z < 0){
            pose.Z = 0;
        }
        pose.ThetaX = req.ThetaX;
        pose.ThetaY = req.ThetaY;
        pose.ThetaZ = req.ThetaZ;

        Eigen::Quaternion<double> g_2;
        g_2 = Eigen::AngleAxis<double>(pose.ThetaX, Eigen::Matrix<double,3,1>::UnitX())
            * Eigen::AngleAxis<double>(pose.ThetaY, Eigen::Matrix<double,3,1>::UnitY())
            * Eigen::AngleAxis<double>(pose.ThetaZ, Eigen::Matrix<double,3,1>::UnitZ());
        if(g_2.w() < 0){
            g_2.w() *= -1;
            g_2.vec() *= -1;
        }
        kst_goal_.add(last_goal_time-start_time_,pose.X,pose.Y,pose.Z,g_2.x(),g_2.y(),g_2.z(),g_2.w());

        return true;
    }

    void updateRobot(const ros::TimerEvent& event){
        cur_time_ = event.current_real.toSec();
        // Query arm for current joint angles
        kinova::KinovaAngles current_angles;
        comm_->getJointAngles(current_angles);

        joints_msg_.position[0] = current_angles.Actuator1 * M_PI/180.0;
        joints_msg_.position[1] = current_angles.Actuator2 * M_PI/180.0;
        joints_msg_.position[2] = current_angles.Actuator3 * M_PI/180.0;
        joints_msg_.position[3] = current_angles.Actuator4 * M_PI/180.0;
        joints_msg_.position[4] = current_angles.Actuator5 * M_PI/180.0;
        joints_msg_.position[5] = current_angles.Actuator6 * M_PI/180.0;

        // Arm finger positions  query
        // kinova::FingerAngles fingers;
        // comm_->getFingerPositions(fingers);
        //
        // joints_msg_.position[6] = fingers.Finger1/6800.0*80.0*M_PI/180.0;
        // joints_msg_.position[7] = fingers.Finger2/6800.0*80.0*M_PI/180.0;
        // joints_msg_.position[8] = fingers.Finger3/6800.0*80.0*M_PI/180.0;
        joints_msg_.position[6] = 0;
        joints_msg_.position[7] = 0;
        joints_msg_.position[8] = 0;

        // Joint velocity
        kinova::KinovaAngles current_vels;
        comm_->getJointVelocities(current_vels);

        joints_msg_.velocity[0] = current_vels.Actuator1;
        joints_msg_.velocity[1] = current_vels.Actuator2;
        joints_msg_.velocity[2] = current_vels.Actuator3;
        joints_msg_.velocity[3] = current_vels.Actuator4;
        joints_msg_.velocity[4] = current_vels.Actuator5;
        joints_msg_.velocity[5] = current_vels.Actuator6;

        // NOTE: Depends on the robot, in this case it always happens
        // Convert the joint velocities from kinova modified velocity type
        for (unsigned int i = 0; i < 6; ++i) {
            double& qd = joints_msg_.velocity[i];
            static const double PI_180 = (math::pi / 180.0);
            // Angle velocities from the API are 0..180 for positive values,
            // and 360..181 for negative ones, in a kind of 2-complement setup.
            if (qd > 180.0) {
                qd -= 360.0;
            }
            qd *= PI_180;
        }

        // Joint torques (effort)
        kinova::KinovaAngles joint_torques;
        comm_->getJointTorques(joint_torques);

        joints_msg_.effort[0] = joint_torques.Actuator1;
        joints_msg_.effort[1] = joint_torques.Actuator2;
        joints_msg_.effort[2] = joint_torques.Actuator3;
        joints_msg_.effort[3] = joint_torques.Actuator4;
        joints_msg_.effort[4] = joint_torques.Actuator5;
        joints_msg_.effort[5] = joint_torques.Actuator6;

        joints_msg_.header.seq++;
        joints_msg_.header.stamp = ros::Time::now();

        // Publish the updated joint state message, mostly for RVIZ
        joints_pub_.publish(joints_msg_);

        kinova::KinovaPose pose_internal;
        comm_->getCartesianPosition(pose_internal);
        // std::cout << pose_internal.X << ", " << pose_internal.Y << ", " << pose_internal.Z << std::endl;
        // std::cout << pose_internal.ThetaX << ", " << pose_internal.ThetaY << ", " << pose_internal.ThetaZ << std::endl;

        // Send the new cartesian position for 5 seconds
        if( cur_time_ < last_goal_time+5.0 ){
            comm_->setCartesianPosition(pose,false);
            // std::cout << pose_internal.ThetaX << ", " << pose_internal.ThetaY << ", " << pose_internal.ThetaZ << std::endl;
        }
        Eigen::Quaternion<double> g_;
        g_ = Eigen::AngleAxis<double>(pose_internal.ThetaX, Eigen::Matrix<double,3,1>::UnitX())
            * Eigen::AngleAxis<double>(pose_internal.ThetaY, Eigen::Matrix<double,3,1>::UnitY())
            * Eigen::AngleAxis<double>(pose_internal.ThetaZ, Eigen::Matrix<double,3,1>::UnitZ());
        if(g_.w() < 0){
            g_.w() *= -1;
            g_.vec() *= -1;
        }
        kst_.add(cur_time_-start_time_,pose_internal.X,pose_internal.Y,pose_internal.Z,g_.x(),g_.y(),g_.z(),g_.w());

    }
};

int main(int argc, char** argv){

    ros::init(argc,argv,"j2n6s300");
    ros::NodeHandle nh_("~");

    SendPose ss(nh_);

    // Spin indefinitly
    ros::spin();

    return 0;
}
