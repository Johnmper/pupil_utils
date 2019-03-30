// #include<stdio>
#include <iostream>
#include <fstream>
#include <utility>
#include <functional>

// ROS INCLUDES
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

// ROS MESSAGE INCLUDES
#include <pupil_msgs/eye_status.h>
#include <pupil_msgs/gaze_datum.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <ros_utils/style.hh>
#include <ros_utils/math.hh>
#include <ros_utils/filter.hh>
#include <ros_utils/beeps.hh>
#include <ros_utils/circular_array.hh>

#undef CV_WINDOW_SHOW
#undef BEEP_5SEC

#ifdef CV_WINDOW_SHOW
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#endif

#define FIXATION_MAX_BUF_SIZE   256

// SyncPolicy Typedef for Readability
typedef message_filters::sync_policies::ExactTime<pupil_msgs::gaze_datum,sensor_msgs::Image> SyncPolicy;

//===========================================================================================================
//   #    #              ######
//   #   #   ####  ##### #     # #      # #    # #    #
//   #  #   #        #   #     # #      # ##   # #   #
//   ###     ####    #   ######  #      # # #  # ####
//   #  #        #   #   #     # #      # #  # # #  #
//   #   #  #    #   #   #     # #      # #   ## #   #
//   #    #  ####    #   ######  ###### # #    # #    #
//===========================================================================================================
struct KstBlink{
    std::string filepath;
    std::ofstream file;

    KstBlink(std::string filepath_) : filepath(filepath_){
        file.open(filepath,std::ofstream::out);
        if(file.fail()){
            std::cerr << "Failed to create " << filepath << " file." << std::endl;
        }else{
            file << "time, is_closed, is_fixating, blink, double_blink, eye_shade, fixation_dispersion, fixation_confidence, confidence";
            file << std::endl;
        }
    }
    ~KstBlink(){
        file.flush();
        file.close();
    }

    bool add(double tm, const pupil_msgs::eye_status& msg){

        if(!file.is_open()){
            std::cerr << "File isn't opened or was prematurely closed" << std::endl;
            return false;
        }

        file << tm << ", ";
        if( msg.is_closed ){ file << 1 << ", ";}
        else{ file << 0 << ", ";}

        if( msg.is_fixating ){ file << 1 << ", ";}
        else{ file << 0 << ", ";}

        if( msg.blink ){ file << 1 << ", ";}
        else{ file << 0 << ", ";}

        if( msg.double_blink ){ file << 1 << ", ";}
        else{ file << 0 << ", ";}

        file << msg.eye_shade << ", ";
        file << msg.fixation_dispersion << ", ";
        file << msg.fixation_confidence << ", ";
        file << msg.confidence;
        file << '\n';

        return true;
    }
};


//===========================================================================================================
//   ######                         #     #
//   #     # #    # #####  # #      #     # ##### # #       ####
//   #     # #    # #    # # #      #     #   #   # #      #
//   ######  #    # #    # # #      #     #   #   # #       ####
//   #       #    # #####  # #      #     #   #   # #           #
//   #       #    # #      # #      #     #   #   # #      #    #
//   #        ####  #      # ######  #####    #   # ######  ####
//===========================================================================================================
namespace pupil_utils
{

//===========================================================================================================
//    #####                        #####  ######
//   #     #   ##   ###### ###### #     # #     #
//   #        #  #      #  #            # #     #
//   #  #### #    #    #   #####   #####  #     #
//   #     # ######   #    #      #       #     #
//   #     # #    #  #     #      #       #     #
//    #####  #    # ###### ###### ####### ######
//===========================================================================================================
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
//   ######                         ######
//   #     # #      # #    # #    # #     # ###### ##### ######  ####  #####  ####  #####
//   #     # #      # ##   # #   #  #     # #        #   #      #    #   #   #    # #    #
//   ######  #      # # #  # ####   #     # #####    #   #####  #        #   #    # #    #
//   #     # #      # #  # # #  #   #     # #        #   #      #        #   #    # #####
//   #     # #      # #   ## #   #  #     # #        #   #      #    #   #   #    # #   #
//   ######  ###### # #    # #    # ######  ######   #   ######  ####    #    ####  #    #
//===========================================================================================================
    class BlinkDetector : public nodelet::Nodelet
    {
    private:
        /// Nodelet Starting time according to ROS TIME
        double start_time;
        /// KST object utility
        KstBlink kst_blink;
        /// Beep utility

        /// Eye is closed flag
        pupil_msgs::eye_status eye_status_msg_;
        /// Publisher warning that eye is closed
        ros::Publisher eye_status_pub_;

        /// RosNodeHandle
        ros::NodeHandlePtr nh_;
        /// Image Transport Object
        std::unique_ptr<image_transport::ImageTransport> eye_it_;
        /// Subscriber with internal Filter for eye image subscriber
        image_transport::SubscriberFilter eye_sub_;
        /// Subscriber with internal filter
        message_filters::Subscriber<pupil_msgs::gaze_datum> gaze_sub_;
        /// Synchronizer!
        std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

        /// Utility array for Gaze Dispersion Calculation
        CircularArray<Gaze2D,FIXATION_MAX_BUF_SIZE>    gaze2d_buf;

        bool in_fixation;
        double fixation_dispersion;
        double fixation_confidence;

        /// Loaded parameters from ROSParam
        int     fixation_buffer_size;
        double  minimum_fixation_duration;
        double  dispersion_threshold;
        double  confidence_threshold;

        double  minimum_blink_duration;
        double  maximum_blink_duration;
        double  minimum_double_blink_wait;
        double  maximum_double_blink_wait;

        int     shade_window_size;
        double  shade_threshold;
        double  shade_lowpass;

        /// Filters
        std::unique_ptr<filter::LowPass<double>> shade_;
        /// Beeps, testing procedure
        audio::Beeps<780> beep;

        bool should_save;
//===========================================================================================================
//    #####
//   #     #  ####  #    #  ####  ##### #####  #    #  ####  #####  ####  #####
//   #       #    # ##   # #        #   #    # #    # #    #   #   #    # #    #
//   #       #    # # #  #  ####    #   #    # #    # #        #   #    # #    #
//   #       #    # #  # #      #   #   #####  #    # #        #   #    # #####
//   #     # #    # #   ## #    #   #   #   #  #    # #    #   #   #    # #   #
//    #####   ####  #    #  ####ifdef    #   #    #  ####   ####    #    ####  #    #
//===========================================================================================================
    public:
        BlinkDetector()
                : kst_blink("/home/johnmper/.ROSData/pupil/kst/blink.txt"){
            #ifdef CV_WINDOW_SHOW
            cv::namedWindow("eye_image",cv::WINDOW_NORMAL);
            cv::resizeWindow("eye_image",400,400);
            #endif

            in_fixation = false;
            fixation_dispersion = 1.0;
            fixation_confidence = 1.0;
            should_save = false;
            ROS_INFO_STREAM(LOG_ID("BlinkDetector")<<"Created");
        }

        virtual
        ~BlinkDetector(){
            #ifdef CV_WINDOW_SHOW
            cv::destroyAllWindows();
            #endif
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
        virtual void
        onInit(){
            nh_.reset(new ros::NodeHandle());
            // Initialization of variables from loaded parameters
            ros::NodeHandle& private_nh_ = getPrivateNodeHandle();

            eye_it_.reset(new image_transport::ImageTransport(*nh_));

            eye_status_pub_ = nh_->advertise<pupil_msgs::eye_status>("/pupil/gaze_0/eye_status",0);

            // Prep synchronizer!
            sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(5), gaze_sub_, eye_sub_));
            sync_->registerCallback(std::bind(&pupil_utils::BlinkDetector::synchronizedCallback, this, std::placeholders::_1, std::placeholders::_2));

            // Subscribe to the desired topics
            eye_sub_.subscribe(*eye_it_, "/pupil/eye_0/image_raw",5);
            if(!eye_sub_.getSubscriber()){
                ROS_ERROR_STREAM(LOG_ID("BlinkDetector") << "Failed to create subscriber to [/pupil/eye_0/image_raw] topic.");
            }

            gaze_sub_.subscribe(*nh_,"/pupil/gaze_0/datum",5);
            if(!gaze_sub_.getSubscriber()){
                ROS_ERROR_STREAM(LOG_ID("BlinkDetector")<<"Failed to create subscriber to [/pupil/gaze_0/datum] topic.");
            }

            fixation_buffer_size = 72;
            minimum_fixation_duration = 0.6;
            dispersion_threshold = 0.001;
            confidence_threshold = 0.8;

            minimum_blink_duration = 0.05;
            maximum_blink_duration = 0.25;
            minimum_double_blink_wait = 0.05;
            maximum_double_blink_wait = 0.5;
            shade_window_size = 7;
            shade_threshold = 0.2;
            shade_lowpass = 0.5;

            private_nh_.getParam("fixation_buffer_size",fixation_buffer_size);
            private_nh_.getParam("minimum_fixation_duration",minimum_fixation_duration);
            private_nh_.getParam("dispersion_threshold",dispersion_threshold);
            private_nh_.getParam("confidence_threshold",confidence_threshold);

            private_nh_.getParam("minimum_blink_duration",minimum_blink_duration);
            private_nh_.getParam("maximum_blink_duration",maximum_blink_duration);
            private_nh_.getParam("minimum_double_blink_wait",minimum_double_blink_wait);
            private_nh_.getParam("maximum_double_blink_wait",maximum_double_blink_wait);
            private_nh_.getParam("shade_window_size",shade_window_size);
            private_nh_.getParam("shade_threshold",shade_threshold);
            private_nh_.getParam("shade_lowpass",shade_lowpass);

            std::cout << " Parameters:\n";
            std::cout << "     fixation_buffer_size: " << fixation_buffer_size << '\n';
            std::cout << "     minimum_fixation_duration: " << minimum_fixation_duration << '\n';
            std::cout << "     dispersion_threshold: " << dispersion_threshold << '\n';
            std::cout << "     confidence_threshold: " << confidence_threshold << '\n';
            std::cout << "     minimum_blink_duration: " << minimum_blink_duration << '\n';
            std::cout << "     maximum_blink_duration: " << maximum_blink_duration << '\n';
            std::cout << "     minimum_double_blink_wait: " << minimum_double_blink_wait << '\n';
            std::cout << "     maximum_double_blink_wait: " << maximum_double_blink_wait << '\n';
            std::cout << "     shade_window_size: " << shade_window_size << '\n';
            std::cout << "     shade_threshold: " << shade_threshold << '\n';
            std::cout << "     shade_lowpass: " << shade_lowpass << '\n';

            shade_.reset(new filter::LowPass<double>(shade_lowpass));

            start_time = ros::Time::now().toSec();
            ROS_INFO_STREAM(LOG_ID("BlinkDetector")<<"Initialization Completed.");
        }


//===========================================================================================================
//    #####                                                                          #####
//   #     # #   # #    #  ####  #    # #####   ####  #    # # ###### ###### #####  #     # #####
//   #        # #  ##   # #    # #    # #    # #    # ##   # #     #  #      #    # #       #    #
//    #####    #   # #  # #      ###### #    # #    # # #  # #    #   #####  #    # #       #####
//         #   #   #  # # #      #    # #####  #    # #  # # #   #    #      #    # #       #    #
//   #     #   #   #   ## #    # #    # #   #  #    # #   ## #  #     #      #    # #     # #    #
//    #####    #   #    #  ####  #    # #    #  ####  #    # # ###### ###### #####   #####  #####
//===========================================================================================================
        void
        synchronizedCallback(const pupil_msgs::gaze_datumConstPtr&  gaze_msg,
                             const sensor_msgs::ImageConstPtr&      eye_msg ){
            // ROS_INFO_STREAM(LOG_ID("BlinkDetector")<<"Synchronized callback called!");
            int u = gaze_msg->base_data[0].ellipse.center.x;
            int v = gaze_msg->base_data[0].ellipse.center.y;

            #ifdef BEEP_5SEC
            static bool already_beeped = false;
            double cur_tm = (gaze_msg->header.stamp.toSec()-start_time);
            double rm = fmod(cur_tm,5.0);
            if(fabs(rm) < 0.1 && !already_beeped){
                already_beeped = true;
                beep.sound.play();
            }
            if( fabs(rm) > 0.02){
                already_beeped = false;
            }
            #endif

            #ifdef CV_WINDOW_SHOW
            // Turn the color image into a CV Matriz, necessary for the
            static cv_bridge::CvImagePtr cv_eye;
            try{
                // No copy made, img_ptr->image to access msg image data directly
                cv_eye = cv_bridge::toCvCopy(eye_msg, sensor_msgs::image_encodings::BGR8);
            }
            catch(cv_bridge::Exception& e ){
                ROS_ERROR("CV_BRIDGE exception: %s", e.what());
                return;
            }
            // std::cout << gaze_msg->base_data[0].ellipse.center.x << ", " << gaze_msg->base_data[0].ellipse.center.y << std::endl;
            if( u >= shade_window_size
                    && u < eye_msg->width-shade_window_size
                    && v >= shade_window_size
                    && v < eye_msg->height-shade_window_size){
                cv::circle(cv_eye->image, cv::Point(u,v), shade_window_size, cv::Scalar(40,40,180), CV_FILLED);
            }
            // Show color image
            cv::imshow("eye_image",cv_eye->image);
            if(cv::waitKey(1) == 's'){
                if(should_save){
                    should_save = false;
                }
                else{
                    should_save = true;
                }
            }
            if(should_save){
                char filename[100];
                sprintf(filename, "/home/johnmper/.ROSData/pupil/eye_frames/seq_%04d.png", eye_msg->header.seq);
                cv::imwrite( filename, cv_eye->image );
            }
            #endif

            // DETECT EYE CLOSED OR OPEN!!
            double shade = 0;
            bool closed = false;
            if( u >= shade_window_size
                    && u < eye_msg->width-shade_window_size
                    && v >= shade_window_size
                    && v < eye_msg->height-shade_window_size){
                int tot = 0;
                uint8_t* T = ((uint8_t*)eye_msg->data.data());
                for(int ii=v-shade_window_size; ii<v+shade_window_size; ++ii){
                    uint8_t* H = &T[3*ii*eye_msg->width];
                    for(int jj=u-shade_window_size; jj<u+shade_window_size; ++jj){
                        shade += (H[3*jj]/3.0 + H[3*jj+1]/3.0 + H[3*jj+2]/3.0);
                        ++tot;
                    }
                }
                shade *= 1.0/(255.0*(double)tot);
                shade = shade_->filter(shade);
                if( shade >= shade_threshold ){
                    closed = true;
                }
            }

            // DETECT 2D GAZE FIXATION (only is eye is opened)
            if( detectFixation(gaze_msg) ){
                if(!in_fixation)
                    std::cout << style::green << "  Fixation Detected!" << style::normal << std::endl;

                in_fixation = true;
            }
            else{
                if(in_fixation)
                    std::cout << style::yellow << "  Not fixated!" << style::normal << std::endl;

                in_fixation = false;
            }


            // DETECT BLINKS
            bool blinked = false;
            bool is_double_blink = false;
            if( in_fixation && (blinked = detectBlink(gaze_msg->header.stamp.toSec()-start_time,closed,is_double_blink)) ){
                if(is_double_blink){
                    std::cout << style::green << "  Double Blink Detected!" << style::normal << std::endl;
                }
                else{
                    std::cout << style::yellow << "  Blink Detected!" << style::normal << std::endl;
                }
            }

            // Update eye_status message
            eye_status_msg_.header = gaze_msg->header;
            eye_status_msg_.is_closed = closed;
            eye_status_msg_.is_fixating = in_fixation;
            eye_status_msg_.blink = blinked;
            eye_status_msg_.double_blink = is_double_blink;
            eye_status_msg_.eye_shade = shade;
            eye_status_msg_.fixation_dispersion = fixation_dispersion;
            eye_status_msg_.fixation_confidence = fixation_confidence;
            // Passthrough from gaze_datum
            eye_status_msg_.confidence = gaze_msg->confidence;
            eye_status_msg_.gaze_normal_3d = gaze_msg->gaze_normal_3d;
            eye_status_msg_.eye_center_3d = gaze_msg->eye_center_3d;

            eye_status_pub_.publish(eye_status_msg_);

            kst_blink.add(eye_status_msg_.header.stamp.toSec()-start_time, eye_status_msg_);
        }


//===========================================================================================================
//   ######                                   #######
//   #     # ###### ##### ######  ####  ##### #       # #    #   ##   ##### #  ####  #    #
//   #     # #        #   #      #    #   #   #       #  #  #   #  #    #   # #    # ##   #
//   #     # #####    #   #####  #        #   #####   #   ##   #    #   #   # #    # # #  #
//   #     # #        #   #      #        #   #       #   ##   ######   #   # #    # #  # #
//   #     # #        #   #      #    #   #   #       #  #  #  #    #   #   # #    # #   ##
//   ######  ######   #   ######  ####    #   #       # #    # #    #   #   #  ####  #    #
//===========================================================================================================
        bool
        detectFixation(const pupil_msgs::gaze_datumConstPtr& gaze){
            Gaze2D new_gaze2d(gaze->header.stamp.toSec()-start_time, gaze->confidence, gaze->norm_pos);
            Gaze2D last_gaze2d = gaze2d_buf.add( new_gaze2d );

            if(gaze2d_buf.is_full()){
                const Gaze2D& dispersion = updateDispersion(gaze2d_buf);
                fixation_dispersion = sqrt(dispersion.point.x*dispersion.point.x + dispersion.point.y*dispersion.point.y);
                fixation_confidence = dispersion.confidence;
                if( fixation_confidence >= confidence_threshold
                        && fixation_dispersion <= dispersion_threshold){
                    return true;
                }
            }

            return false;
        }



//===========================================================================================================
//   #     #                                   ######
//   #     # #####  #####    ##   ##### ###### #     # #  ####  #####  ###### #####   ####  #  ####  #    #
//   #     # #    # #    #  #  #    #   #      #     # # #      #    # #      #    # #      # #    # ##   #
//   #     # #    # #    # #    #   #   #####  #     # #  ####  #    # #####  #    #  ####  # #    # # #  #
//   #     # #####  #    # ######   #   #      #     # #      # #####  #      #####       # # #    # #  # #
//   #     # #      #    # #    #   #   #      #     # # #    # #      #      #   #  #    # # #    # #   ##
//    #####  #      #####  #    #   #   ###### ######  #  ####  #      ###### #    #  ####  #  ####  #    #
//===========================================================================================================
        const Gaze2D&
        updateDispersion(CircularArray<Gaze2D,FIXATION_MAX_BUF_SIZE>& buffer){
            static Gaze2D var_gaze;
            static pupil_msgs::point2d mean_2d, var_2d;

            // Pass through every element of Circular Array Buffer and calculate the necessary parameters
            double tot_ = 0;
            double max_ = 0;
            Gaze2D& cur_gaze = buffer[0];
            std::size_t sz = buffer.size();
            mean_2d.x = 0; mean_2d.y = 0;
            var_2d.x = 0; var_2d.y = 0;
            // Calculate the mean 2d gaze position
            for(std::size_t ii=0; ii<sz; ++ii){
                Gaze2D& gaze_ = buffer[ii];
                if( gaze_.tm < cur_gaze.tm-minimum_fixation_duration ){
                    break;
                }
                tot_ += gaze_.confidence;
                max_ += 1.0;
                mean_2d.x += (gaze_.point.x*gaze_.confidence);
                mean_2d.y += (gaze_.point.y*gaze_.confidence);
            }
            if( tot_ == 0){
                var_gaze.confidence = 0;
                var_gaze.point = var_2d;
                return var_gaze;
            }else{
                mean_2d.x /= tot_;
                mean_2d.y /= tot_;
            }

            // calculate the gaze dispersion
            for(std::size_t ii=0; ii<sz; ++ii){
                Gaze2D& gaze_ = buffer[ii];
                if( gaze_.tm < cur_gaze.tm-minimum_fixation_duration ){
                    break;
                }
                var_2d.x += (gaze_.point.x - mean_2d.x)*(gaze_.point.x - mean_2d.x)*gaze_.confidence;
                var_2d.y += (gaze_.point.y - mean_2d.y)*(gaze_.point.y - mean_2d.y)*gaze_.confidence;
            }
            var_2d.x = sqrt(var_2d.x/tot_);
            var_2d.y = sqrt(var_2d.y/tot_);

            var_gaze.confidence = tot_/max_;
            var_gaze.point = var_2d;
            return var_gaze;
        }

//===========================================================================================================
//   ######                                   ######
//   #     # ###### ##### ######  ####  ##### #     # #      # #    # #    #
//   #     # #        #   #      #    #   #   #     # #      # ##   # #   #
//   #     # #####    #   #####  #        #   ######  #      # # #  # ####
//   #     # #        #   #      #        #   #     # #      # #  # # #  #
//   #     # #        #   #      #    #   #   #     # #      # #   ## #   #
//   ######  ######   #   ######  ####    #   ######  ###### # #    # #    #
//===========================================================================================================
        bool
        detectBlink(double tm, bool closed, bool& is_double_blink){
            static bool last_status = false;
            static int blink_count = 0;
            static double time_between_blinks = 0;
            static double last_open_tm = 0;
            static double last_closed_tm = 0;

            bool detected_blink = false;
            // Transition
            if(last_status != closed){
                // std::cout << "## " << tm<<": ";
                if(closed){
                    // Transition from Open -> Closed
                    last_open_tm = tm;
                    time_between_blinks = last_open_tm-last_closed_tm;
                    if(time_between_blinks < minimum_double_blink_wait
                            || time_between_blinks > maximum_double_blink_wait ){
                        blink_count = 0;
                    }
                    // std::cout << "<" << time_between_blinks << ">\n" << blink_count;
                }
                else{
                    // Transition from Closed -> Open
                    last_closed_tm = tm;
                    double blink_duration = last_closed_tm-last_open_tm;
                    // std::cout << ">" << blink_duration << "<\n";
                    if( blink_duration > minimum_blink_duration
                            && blink_duration < maximum_blink_duration ){
                        blink_count++;
                        detected_blink = true;
                        // std::cout << blink_count << " |> " << style::green << "Valid" << style::normal;
                    }
                    else{
                        blink_count = 0;
                        // std::cout << blink_count << " |> " << style::yellow << "Invalid" << style::normal;
                    }
                }
                std::cout << std::endl;
            }

            // Check if detected blink is double blink or normal blink
            if(detected_blink && blink_count == 2){
                is_double_blink = true;
                blink_count = 0;
            }
            else{
                is_double_blink = false;
            }

            last_status = closed;
            return detected_blink;
        }

    };

    PLUGINLIB_DECLARE_CLASS(pupil_utils, BlinkDetector, pupil_utils::BlinkDetector, nodelet::Nodelet);
}
