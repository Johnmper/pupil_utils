
#include <librealsense/rs.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

int main(int argc, char** argv) try
{
    ros::init(argc,argv,"save_camera");
    rs::log_to_console(rs::log_severity::warn);
    //rs::log_to_file(rs::log_severity::debug, "librealsense.log");

    // Obtain a list of devices currently present on the system
    rs::context ctx;
    int device_count = ctx.get_device_count();
    if (!device_count) printf("No device detected. Is it plugged in?\n");

    nlohmann::json info_json;

    std::ofstream r200_info("/home/johnmper/.ROSData/pupil/r200_info.json",std::ofstream::out);

    // Show the device name and information
    rs::device * dev = ctx.get_device(0);

    info_json["device_0"]["name"] = dev->get_name();
    info_json["device_0"]["serial_number"] = dev->get_serial();
    info_json["device_0"]["firmware_version"] = dev->get_firmware_version();

    for (int j = RS_CAMERA_INFO_DEVICE_NAME; j < RS_CAMERA_INFO_COUNT; ++j)
    {
        rs::camera_info param = (rs::camera_info)j;
        if (dev->supports(param)){
            info_json["device_0"]["camera_info"][rs_camera_info_to_string(rs_camera_info(param))] = dev->get_info(param);
        }
    }

    for(int j = 0; j < RS_OPTION_COUNT; ++j)
    {
        rs::option opt = (rs::option)j;
        if(dev->supports_option(opt))
        {
            std::stringstream ss;
            double min, max, step, def;
            dev->get_option_range(opt, min, max, step, def);
            ss << opt;
            info_json["device_0"]["options"][ss.str()]["default"] = def;
            info_json["device_0"]["options"][ss.str()]["min"] = min;
            info_json["device_0"]["options"][ss.str()]["max"] = max;
            info_json["device_0"]["options"][ss.str()]["step"] = step;
        }
    }

    for(int j = 0; j < RS_STREAM_COUNT; ++j){

        rs::stream strm = (rs::stream)j;

        int mode_count = dev->get_stream_mode_count(strm);
        if(mode_count == 0) continue;

        if(strm == rs::stream::infrared) continue;
        else if(strm == rs::stream::infrared2) continue;
        else if(strm == rs::stream::depth){

            for(int k = 0; k < mode_count; ++k){
                int width, height, framerate;
                rs::format format;
                dev->get_stream_mode(strm, k, width, height, format, framerate);
                dev->enable_stream(strm, width, height, format, framerate);
                rs::intrinsics intrin = dev->get_stream_intrinsics(strm);

                info_json["device_0"]["depth"][std::to_string(width)+"x"+std::to_string(height)]["FoV"]["H"] = intrin.hfov();
                info_json["device_0"]["depth"][std::to_string(width)+"x"+std::to_string(height)]["FoV"]["V"] = intrin.vfov();
                info_json["device_0"]["depth"][std::to_string(width)+"x"+std::to_string(height)]["D"] = intrin.coeffs;
                info_json["device_0"]["depth"][std::to_string(width)+"x"+std::to_string(height)]["fx"] = intrin.fx;
                info_json["device_0"]["depth"][std::to_string(width)+"x"+std::to_string(height)]["fy"] = intrin.fy;
                info_json["device_0"]["depth"][std::to_string(width)+"x"+std::to_string(height)]["ppx"] = intrin.ppx;
                info_json["device_0"]["depth"][std::to_string(width)+"x"+std::to_string(height)]["ppy"] = intrin.ppy;
                info_json["device_0"]["depth"][std::to_string(width)+"x"+std::to_string(height)]["width"] = intrin.width;
                info_json["device_0"]["depth"][std::to_string(width)+"x"+std::to_string(height)]["height"] = intrin.height;
            }

        }
        else if(strm == rs::stream::color){

            for(int k = 0; k < mode_count; ++k){
                int width, height, framerate;
                rs::format format;
                dev->get_stream_mode(strm, k, width, height, format, framerate);
                dev->enable_stream(strm, width, height, format, framerate);
                rs::intrinsics intrin = dev->get_stream_intrinsics(strm);

                info_json["device_0"]["color"][std::to_string(width)+"x"+std::to_string(height)]["FoV"]["H"] = intrin.hfov();
                info_json["device_0"]["color"][std::to_string(width)+"x"+std::to_string(height)]["FoV"]["V"] = intrin.vfov();
                info_json["device_0"]["color"][std::to_string(width)+"x"+std::to_string(height)]["D"] = intrin.coeffs;
                info_json["device_0"]["color"][std::to_string(width)+"x"+std::to_string(height)]["fx"] = intrin.fx;
                info_json["device_0"]["color"][std::to_string(width)+"x"+std::to_string(height)]["fy"] = intrin.fy;
                info_json["device_0"]["color"][std::to_string(width)+"x"+std::to_string(height)]["ppx"] = intrin.ppx;
                info_json["device_0"]["color"][std::to_string(width)+"x"+std::to_string(height)]["ppy"] = intrin.ppy;
                info_json["device_0"]["color"][std::to_string(width)+"x"+std::to_string(height)]["width"] = intrin.width;
                info_json["device_0"]["color"][std::to_string(width)+"x"+std::to_string(height)]["height"] = intrin.height;

            }

        }

        dev->disable_stream(strm);
    }


    const std::string DEFAULT_BASE_FRAME_ID = "r200_camera_link";
    const std::string DEFAULT_DEPTH_FRAME_ID = "r200_depth_frame";
    const std::string DEFAULT_COLOR_FRAME_ID = "r200_color_frame";
    const std::string DEFAULT_DEPTH_OPTICAL_FRAME_ID = "depth_optical_frame";
    const std::string DEFAULT_COLOR_OPTICAL_FRAME_ID = "rgb_optical_frame";

    // Publish transforms for the cameras
    tf::Quaternion q_c2co;
    tf::Quaternion q_d2do;
    tf::Quaternion q_i2io;

    info_json["device_0"]["tf"][DEFAULT_COLOR_FRAME_ID]["parent"] = DEFAULT_BASE_FRAME_ID;
    info_json["device_0"]["tf"][DEFAULT_COLOR_FRAME_ID]["T"]["x"] = 0;
    info_json["device_0"]["tf"][DEFAULT_COLOR_FRAME_ID]["T"]["y"] = 0;
    info_json["device_0"]["tf"][DEFAULT_COLOR_FRAME_ID]["T"]["z"] = 0;
    info_json["device_0"]["tf"][DEFAULT_COLOR_FRAME_ID]["R"]["x"] = 0;
    info_json["device_0"]["tf"][DEFAULT_COLOR_FRAME_ID]["R"]["y"] = 0;
    info_json["device_0"]["tf"][DEFAULT_COLOR_FRAME_ID]["R"]["z"] = 0;
    info_json["device_0"]["tf"][DEFAULT_COLOR_FRAME_ID]["R"]["w"] = 1;

    // Transform color frame to color optical frame
    q_c2co.setRPY(-M_PI / 2, 0.0, -M_PI / 2);
    info_json["device_0"]["tf"][DEFAULT_COLOR_OPTICAL_FRAME_ID]["parent"] = DEFAULT_COLOR_FRAME_ID;
    info_json["device_0"]["tf"][DEFAULT_COLOR_OPTICAL_FRAME_ID]["T"]["x"] = 0;
    info_json["device_0"]["tf"][DEFAULT_COLOR_OPTICAL_FRAME_ID]["T"]["y"] = 0;
    info_json["device_0"]["tf"][DEFAULT_COLOR_OPTICAL_FRAME_ID]["T"]["z"] = 0;
    info_json["device_0"]["tf"][DEFAULT_COLOR_OPTICAL_FRAME_ID]["R"]["x"] = q_c2co.getX();
    info_json["device_0"]["tf"][DEFAULT_COLOR_OPTICAL_FRAME_ID]["R"]["y"] = q_c2co.getY();
    info_json["device_0"]["tf"][DEFAULT_COLOR_OPTICAL_FRAME_ID]["R"]["z"] = q_c2co.getZ();
    info_json["device_0"]["tf"][DEFAULT_COLOR_OPTICAL_FRAME_ID]["R"]["w"] = q_c2co.getW();

    // Transform base frame to depth frame
    rs::device* device = ctx.get_device(0);
    rs::extrinsics color2depth_extrinsic = device->get_extrinsics(rs::stream::color, rs::stream::depth);
    info_json["device_0"]["tf"][DEFAULT_DEPTH_FRAME_ID]["parent"] = DEFAULT_BASE_FRAME_ID;
    info_json["device_0"]["tf"][DEFAULT_DEPTH_FRAME_ID]["T"]["x"] = color2depth_extrinsic.translation[2];
    info_json["device_0"]["tf"][DEFAULT_DEPTH_FRAME_ID]["T"]["y"] = -color2depth_extrinsic.translation[0];
    info_json["device_0"]["tf"][DEFAULT_DEPTH_FRAME_ID]["T"]["z"] = -color2depth_extrinsic.translation[1];
    info_json["device_0"]["tf"][DEFAULT_DEPTH_FRAME_ID]["R"]["x"] = 0;
    info_json["device_0"]["tf"][DEFAULT_DEPTH_FRAME_ID]["R"]["y"] = 0;
    info_json["device_0"]["tf"][DEFAULT_DEPTH_FRAME_ID]["R"]["z"] = 0;
    info_json["device_0"]["tf"][DEFAULT_DEPTH_FRAME_ID]["R"]["w"] = 1;


    // Transform depth frame to depth optical frame
    q_d2do.setRPY(-M_PI / 2, 0.0, -M_PI / 2);
    info_json["device_0"]["tf"][DEFAULT_DEPTH_OPTICAL_FRAME_ID]["parent"] = DEFAULT_DEPTH_FRAME_ID;
    info_json["device_0"]["tf"][DEFAULT_DEPTH_OPTICAL_FRAME_ID]["T"]["x"] = 0;
    info_json["device_0"]["tf"][DEFAULT_DEPTH_OPTICAL_FRAME_ID]["T"]["y"] = 0;
    info_json["device_0"]["tf"][DEFAULT_DEPTH_OPTICAL_FRAME_ID]["T"]["z"] = 0;
    info_json["device_0"]["tf"][DEFAULT_DEPTH_OPTICAL_FRAME_ID]["R"]["x"] = q_d2do.getX();
    info_json["device_0"]["tf"][DEFAULT_DEPTH_OPTICAL_FRAME_ID]["R"]["y"] = q_d2do.getY();
    info_json["device_0"]["tf"][DEFAULT_DEPTH_OPTICAL_FRAME_ID]["R"]["z"] = q_d2do.getZ();
    info_json["device_0"]["tf"][DEFAULT_DEPTH_OPTICAL_FRAME_ID]["R"]["w"] = q_d2do.getW();

    r200_info << std::setw(2) << info_json;

    return EXIT_SUCCESS;
}
catch(const rs::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
