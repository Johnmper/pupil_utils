// STANDARD C++ INCLUDES
#include<iostream>

// ROS INCLUDES
#include<ros/ros.h>
#include<image_transport/image_transport.h>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/highgui/highgui.hpp>
#include<sensor_msgs/image_encodings.h>

// OPENCV INCLUDES
#include<Eigen/Eigen> // Full eigen library
#include<opencv2/core/core.hpp>
#include<opencv2/core/eigen.hpp>

// LOCAL INCLUDES



#define WINDOW_NAME_RGB "Depth image viewer"
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#define NODE_NAME   "viewer_depth"

#define SAVE_FOLDER "/home/johnmper/Thesis/Data/r200_depth/d2750/"
unsigned int kk = 0;
bool recording = false;

float fx = 589.7913818359375;
float fy = 589.7913818359375;
float center_x = 337.5098876953125;
float center_y = 239.5;
float constant_x = 1.0/fx;
float constant_y = 1.0/fy;

/// Eigen value and vetors solver
Eigen::EigenSolver<Eigen::Matrix<double,3,3>> eig_solver;

void imgRecvCallback(const sensor_msgs::ImageConstPtr& msg){
    static cv_bridge::CvImageConstPtr img_ptr;

    try{
        // No copy made, img_ptr->image to access msg image data directly
        img_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        // Alternative encoding for Depth Stream:
    }
    catch(cv_bridge::Exception& e ){
        ROS_ERROR("CV_BRIDGE exception: %s", e.what());
        return;
    }

    if(recording){
        std::string filename;
        if(kk < 10){
            filename = std::string(SAVE_FOLDER)+std::string("seq_0")+std::to_string(kk)+std::string(".png");
        }
        else if(kk < 100){
            filename = std::string(SAVE_FOLDER)+std::string("seq_")+std::to_string(kk)+std::string(".png");
        }

        if(kk < 100){
            std::cout << "Saving frame number " << kk << std::endl;
            cv::imwrite( filename.c_str(), img_ptr->image );
        }
        ++kk;
    }
    else{

        double n_points = 0;
        Eigen::Matrix<double,3,1> mean_point;
        mean_point << 0,0,0;
        uint16_t* D = ((uint16_t*)msg->data.data());

        // GET CENTROID
        for(uint32_t ii=0; ii < msg->height; ++ii)
        {
            uint16_t* R = &D[ii*msg->width];
            for(uint32_t jj=0; jj < msg->width; ++jj)
            {
                if( 710U < R[jj] && R[jj] < 3500U){
                    double depth = R[jj]*0.001;

                    Eigen::Matrix<double,3,1> point;
                    point << (jj - center_x) * depth * constant_x, (ii - center_y) * depth * constant_y, depth;
                    mean_point += point;
                    n_points += 1.0;
                }
            }
        }
        mean_point = mean_point/n_points;

        std::cout << mean_point << std::endl;
        std::cout << n_points << std::endl;
        // GET COVARIANCE
        Eigen::Matrix<double,3,3> C;
        C << 0,0,0,0,0,0,0,0,0;
        for(uint32_t ii=0; ii < msg->height; ++ii)
        {
            uint16_t* R = &D[ii*msg->width];
            for(uint32_t jj=0; jj < msg->width; ++jj)
            {
                if( 710U < R[jj] && R[jj] < 3500U){
                    double depth = R[jj]*0.001;

                    Eigen::Matrix<double,3,1> point;
                    point << (jj - center_x) * depth * constant_x, (ii - center_y) * depth * constant_y, depth;
                    point = mean_point-point;

                    C.coeffRef(0,0) += point.coeffRef(0)*point.coeffRef(0);
                    C.coeffRef(1,1) += point.coeffRef(1)*point.coeffRef(1);
                    C.coeffRef(2,2) += point.coeffRef(2)*point.coeffRef(2);
                    C.coeffRef(0,1) += point.coeffRef(0)*point.coeffRef(1);
                    C.coeffRef(0,2) += point.coeffRef(0)*point.coeffRef(2);
                    C.coeffRef(1,2) += point.coeffRef(1)*point.coeffRef(2);
                }
            }
        }
        // Fill the rest of the covariance matrix
        C.coeffRef(1,0) = C.coeffRef(0,1);
        C.coeffRef(2,0) = C.coeffRef(0,2);
        C.coeffRef(2,1) = C.coeffRef(1,2);

        // Calculate the Eigen vectors and eigen values, reorder vectors according to the diagonal values
        eig_solver.compute(C,true);
        Eigen::Matrix<std::complex<double>,3,1> val = eig_solver.eigenvalues();
        Eigen::Matrix<std::complex<double>,3,3> vec = eig_solver.eigenvectors();

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

        std::cout << R.col(2).transpose() << std::endl;
    }


    cv::imshow(WINDOW_NAME_RGB,img_ptr->image);

    if(cv::waitKey(3) == 's'){
        if(!recording){
            kk = 0;
            recording = true;
            std::cout << "Starting Recording" << std::endl;
        }
        else{
            recording = false;
        }
    }
}


int main(int argc, char** argv){

    ros::init(argc,argv,NODE_NAME);

    ros::NodeHandle node;
    image_transport::Subscriber img_sub;
    image_transport::ImageTransport it_trp(node);

    img_sub = it_trp.subscribe("/pupil/depth/image_raw", 1, &imgRecvCallback);

    ros::spin();

    return 0;
}
