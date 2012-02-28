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

using std::vector;
using pf::ParticleFilter;

/* Classes */
class VehicleFeatures {
private:
    float mid_dist;
public:
    vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    vector< float > rank;
    cv::FlannBasedMatcher matcher;

    VehicleFeatures();

    void findPairs(cv::Mat& sceneDescriptors, vector<cv::DMatch>& pairs, vector<cv::DMatch>& negpairs);

};

class VehicleDetector {
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;

    unsigned long long frame_count;
    int64 tt;

    VehicleFeatures referenceCar;

    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;

    ParticleFilter pf;

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

