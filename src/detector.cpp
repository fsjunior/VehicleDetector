#include "detector.h"
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>


using std::vector;


static const char WINDOW[] = "Vehicle Detector";
static const char REFERENCE_CAR[] = "images/carro.png";
static const float KEYPOINTS_FACTOR = 0.3;
static const unsigned int DYNAMIC_FEATURE_COUNT = 10;

/* Functions */
bool compKeyPoint(const cv::KeyPoint& a, const cv::KeyPoint& b)
{
    return a.response > b.response;
}

float sigmoid(float x)
{
    return std::tanh(x);
}

VehicleFeatures::VehicleFeatures()
: mid_dist(0.)//, matcher(new cv::flann::KDTreeIndexParams(4))
{
};

void VehicleFeatures::findPairs(cv::Mat& sceneDescriptors, vector<cv::DMatch>& pairs, vector<cv::DMatch>& negpairs)
{
    vector< cv::DMatch > matches;
    float min_dist, max_dist, lmid_dist;

    matcher.match(descriptors, sceneDescriptors, matches);

    min_dist = max_dist = matches.begin()->distance;
    lmid_dist = 0.0;

    for (vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
        if (i->distance < min_dist)
            min_dist = i->distance;
        if (i->distance > max_dist)
            max_dist = i->distance;
        lmid_dist += i->distance;

    }

    lmid_dist = lmid_dist / (float) matches.size();

    if (mid_dist == 0.)
        mid_dist = lmid_dist;
    else
        mid_dist = (mid_dist * 1.95 + lmid_dist * 0.05) / 2.;
    ROS_DEBUG("mid_dist: %f; local_mid_dist %f", mid_dist, lmid_dist);
    ROS_DEBUG("Min/Max dist: %f %f", min_dist, max_dist);

    for (vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
        rank[i->queryIdx] = sigmoid(rank[i->queryIdx] + -(i->distance - mid_dist)*2.0);
        if (rank[i->queryIdx] >= 0.0)
            pairs.push_back(*i);
        else
            negpairs.push_back(*i);
    }
}

VehicleDetector::VehicleDetector()
: it_(nh_), frame_count(0), tt(0)
{
    image_sub_ = it_.subscribe("/stereo/left/image_rect_mono", 10, &VehicleDetector::imageCb, this);

    detector = new cv::SurfFeatureDetector(400., 3, 4, true);
    extractor = new cv::SurfDescriptorExtractor(3, 4, false, true);

    if (detector.empty() || extractor.empty()) {
        ROS_ERROR("Error trying to initialize SURF Descipror.");
        ros::shutdown();
    }

    if (!loadReferenceCar()) {
        ROS_ERROR("Error opening reference car in: %s.", REFERENCE_CAR);
        ros::shutdown();
    }

    cv::namedWindow(WINDOW);
}

VehicleDetector::~VehicleDetector()
{
    cv::destroyWindow(WINDOW);
}

void VehicleDetector::drawKeyPoints(cv::Mat& image, vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keyPoints, const cv::Scalar& color, int thickness)
{
    vector<cv::KeyPoint>::iterator k;

    for (vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
        k = keyPoints.begin() + i->trainIdx;
        cv::circle(image, k->pt, k->size * 1.2 / 9. * 2, color, thickness);
    }
}

cv::Point2f VehicleDetector::getVehicleCentralPoint(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &sceneKeyPoints)
{
    vector<cv::Point2f> src, dst;
    vector<cv::Point2f> srcpt, dstpt;

    for (vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
        src.push_back(referenceCar.keyPoints[i->queryIdx].pt);
        dst.push_back(sceneKeyPoints[i->trainIdx].pt);
    }

    cv::Mat h = cv::findHomography(src, dst, cv::RANSAC);

    srcpt.push_back(cv::Point2f(96, 42.));

    cv::perspectiveTransform(srcpt, dstpt, h);

    return dstpt[0];
}

void VehicleDetector::imageCb(const sensor_msgs::ImageConstPtr& msg)
{
    int64 tl;

    tl = cv::getTickCount();

    cv_bridge::CvImagePtr cv_ptr;
    vector<cv::KeyPoint> sceneKeypoints;
    cv::Mat sceneDescriptors;
    vector<cv::DMatch> pairs, negpairs, dyn_pairs, dyn_negpairs;


    try {
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());

        return;
    }

    detector->detect(cv_ptr->image, sceneKeypoints);
    extractor->compute(cv_ptr->image, sceneKeypoints, sceneDescriptors);

    referenceCar.findPairs(sceneDescriptors, pairs, negpairs);
    try {
        cv::Point2f p = getVehicleCentralPoint(pairs, sceneKeypoints);
        ROS_INFO("%f %f", p.x, p.y);
        cv::circle(cv_ptr->image, p, 2, cv::Scalar(0, 255, 255), 3);
    } catch (cv::Exception e) {
        //ROS_ERROR_STREAM(e.err);
    }

    drawKeyPoints(cv_ptr->image, pairs, sceneKeypoints, cv::Scalar(0, 255, 0), 2);
    drawKeyPoints(cv_ptr->image, negpairs, sceneKeypoints, cv::Scalar(0, 0, 255), 1);

    //pf.update(pairs, sceneKeypoints);

    //pf.printParticles(cv_ptr->image);

    cv::imshow(WINDOW, cv_ptr->image);

    frame_count++;

    tt = tt + (cv::getTickCount() - tl);

    cv::waitKey(3);
}

bool VehicleDetector::loadReferenceCar()
{
    cv::Mat referenceCarImage;
    referenceCarImage = cv::imread(REFERENCE_CAR, CV_LOAD_IMAGE_GRAYSCALE);
    if (referenceCarImage.data) {
        detector->detect(referenceCarImage, referenceCar.keyPoints);
        ROS_INFO("Keypoints in object: %d; Selecting the best %d", (int)referenceCar.keyPoints.size(), (int) ((float) referenceCar.keyPoints.size() * KEYPOINTS_FACTOR));
        std::sort(referenceCar.keyPoints.begin(), referenceCar.keyPoints.end(), compKeyPoint);
        referenceCar.keyPoints.resize((int) ((float) referenceCar.keyPoints.size() * KEYPOINTS_FACTOR));
        referenceCar.rank.resize(referenceCar.keyPoints.size());

        extractor->compute(referenceCarImage, referenceCar.keyPoints, referenceCar.descriptors);

        return true;
    }
    return false;
}

float VehicleDetector::getFPS()
{
    return frame_count / ((float) tt / (cv::getTickFrequency()));
}