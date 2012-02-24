/* 
 * File:   main.cpp
 * Author: linuxer
 *
 * Created on February 24, 2012, 12:28 AM
 */

#include "detector.h"
#include <iostream>



int main(int argc, char** argv)
{
    ros::init(argc, argv, "vehicle_detector");

    VehicleDetector vd;
    ros::spin();

    std::cout << "Frames por segundo: " << vd.getFPS() << std::endl;
    return 0;
}
