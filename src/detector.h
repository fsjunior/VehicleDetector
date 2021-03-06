/* 
 * File:   detector.h
 * Author: linuxer
 *
 * Created on February 24, 2012, 12:26 AM
 */

#ifndef DETECTOR_H
#define	DETECTOR_H
#include "particlefilter.h"
#include <image_transport/image_transport.h>
#include <boost/random.hpp>
#include <fstream>


using std::vector;
using pf::ParticleFilter;

/* Classes */
class VehicleDetector;

class DataLogger {
    boost::mt19937 rng;
    boost::uniform_int<> maxRand;
    //boost::variate_generator< boost::mt19937&, boost::uniform_int<> > genRand;


    std::ofstream dataLog;
public:
    int status;

    bool homographyDetected;
    bool homographyComputed;
    cv::Point2f homographyCenter;

    bool particleFilterDetected;
    cv::Point2f particleFilterCenter;
    float particleFilterStdDev;

    
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > genRand;

    DataLogger(const std::string &filename_);
    ~DataLogger();
    void saveData();
};

class VehicleFeatures {
private:
    float mid_dist;
public:
    vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    vector< float > rank;
    //cv::
    cv::FlannBasedMatcher matcher;

    VehicleFeatures();

    void findPairs(cv::Mat& sceneDescriptors, vector<cv::DMatch>& pairs, vector<cv::DMatch>& negpairs);

};

class VehicleDetector {
    cv::Mat referenceCarImage;
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;

    unsigned long long frame_count;
    int64 tt;

    VehicleFeatures referenceCar;

    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;

    ParticleFilter pf;

    DataLogger dl;

    

public:
    VehicleDetector();

    ~VehicleDetector();

    void drawKeyPoints(cv::Mat& image, vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keyPoints, const cv::Scalar& color, int thickness);

    cv::Point2f getVehicleCentralPoint(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &sceneKeyPoints);



    void imageCb(const sensor_msgs::ImageConstPtr& msg);

    bool loadReferenceCar();

    float getFPS();
};





#endif	/* DETECTOR_H */

