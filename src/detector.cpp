#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/miniflann.hpp>
//#include <cv.h>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace enc = sensor_msgs::image_encodings;
using std::vector;


static const char WINDOW[] = "Vehicle Detector";
//static const char NEGWINDOW[] = "neg-Vehicle Detector";
static const char REFERENCE_CAR[] = "images/carro.png";
static const float KEYPOINTS_FACTOR = 0.3;

class VehicleFeatures {
public:
    cv::Mat vehicleImage;

    vector<cv::KeyPoint> refCarKeypoints;
    cv::Mat refCarDescriptors;
    vector< float > prob;
};

bool compKeyPoint(const cv::KeyPoint& a, const cv::KeyPoint& b)
{
    return a.response > b.response;
}

float sigmoid(float x)
{
    return std::tanh(x);
}

class VehicleDetector {
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;

    VehicleFeatures referenceCar;


    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    //cv::SURF surf(500, 4, 2, false, true);

public:

    VehicleDetector()
    : it_(nh_)
    {
        image_sub_ = it_.subscribe("/stereo/left/image_rect_mono", 10, &VehicleDetector::imageCb, this);

        detector = new cv::SurfFeatureDetector(400., 3, 4, true);
        //detector = new cv::SurfFeatureDetector();

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
        //cv::namedWindow(NEGWINDOW);
    }

    ~VehicleDetector()
    {

        cv::destroyWindow(WINDOW);
        //cv::destroyWindow(NEGWINDOW);

        //surf.delete_obj();
    }

    void findPairs(cv::Mat& sceneDescriptors, vector<cv::DMatch>& pairs, vector<cv::DMatch>& negpairs)
    {

        cv::FlannBasedMatcher matcher;
        vector< cv::DMatch > matches;
        float min_dist, max_dist, mid_dist;

        matcher.match(referenceCar.refCarDescriptors, sceneDescriptors, matches);

        min_dist = max_dist = matches.begin()->distance;
        mid_dist = 0.0;

        for (vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
            if (i->distance < min_dist)
                min_dist = i->distance;
            if (i->distance > max_dist)
                max_dist = i->distance;
            mid_dist += i->distance;

        }

        mid_dist = mid_dist / (float) matches.size();

        ROS_DEBUG("Min/Max/Mid dist: %f %f %f", min_dist, max_dist, mid_dist);

        for (vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
            referenceCar.prob[i->queryIdx] = sigmoid(referenceCar.prob[i->queryIdx] + -(i->distance - mid_dist)*2.0);
            ROS_DEBUG("Prob: [%d]: %f", i->queryIdx, referenceCar.prob[i->queryIdx]);
            if (referenceCar.prob[i->queryIdx] >= 0.0)
                pairs.push_back(*i);
            else
                negpairs.push_back(*i);
        }







        /*for(vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
            if(i->distance < min_dist)
                min_dist = i->distance;
            ROS_INFO("queryidx: %d\n", i->queryIdx);
        }


        for(vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++)
            if(i->distance <= min_dist*1.5)
                pairs.push_back(*i);*/


    }

    void drawKeyPoints(cv::Mat& image, vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints, const cv::Scalar& color, int thickness)
    {
        vector<cv::KeyPoint>::iterator k;

        for (vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
            k = keypoints.begin() + i->trainIdx;
            cv::circle(image, k->pt, k->size * 1.2 / 9. * 2, color, thickness);
        }
    }

    void imageCb(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        vector<cv::KeyPoint> sceneKeypoints;
        cv::Mat sceneDescriptors;
        vector<cv::DMatch> pairs, negpairs;

        try {
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        detector->detect(cv_ptr->image, sceneKeypoints);
        extractor->compute(cv_ptr->image, sceneKeypoints, sceneDescriptors);


        findPairs(sceneDescriptors, pairs, negpairs);

        

        drawKeyPoints(cv_ptr->image, pairs, sceneKeypoints, cv::Scalar(0, 255, 0), 2);
        drawKeyPoints(cv_ptr->image, negpairs, sceneKeypoints, cv::Scalar(0, 0, 255), 1);


        cv::imshow(WINDOW, cv_ptr->image);


        //for (vector<cv::KeyPoint>::iterator k = sceneKeypoints.begin(); k != sceneKeypoints.end(); k++)
        //    cv::circle(cv_ptr->image, k->pt, k->size * 1.2 / 9. * 2, cv::Scalar(255));

        //cv::imshow(WINDOW, cv_ptr->image);
        cv::waitKey(3);

    }

    bool loadReferenceCar()
    {
        referenceCar.vehicleImage = cv::imread(REFERENCE_CAR, CV_LOAD_IMAGE_GRAYSCALE);
        if (referenceCar.vehicleImage.data) {
            detector->detect(referenceCar.vehicleImage, referenceCar.refCarKeypoints);

            std::sort(referenceCar.refCarKeypoints.begin(), referenceCar.refCarKeypoints.end(), compKeyPoint);


            ROS_INFO("Keypoints in object: %d; Selecting the best %d", referenceCar.refCarKeypoints.size(), (int) ((float) referenceCar.refCarKeypoints.size() * KEYPOINTS_FACTOR));


            referenceCar.refCarKeypoints.resize((int) ((float) referenceCar.refCarKeypoints.size() * KEYPOINTS_FACTOR));



            extractor->compute(referenceCar.vehicleImage, referenceCar.refCarKeypoints, referenceCar.refCarDescriptors);

            referenceCar.prob.resize(referenceCar.refCarKeypoints.size());

            //(*surf)(referenceCar, mask, refCarKeypoints, refCarDescriptors); //Calcula keypoints e descriptors

            return true;
        }
        return false;
    }





};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vehicle_detector");

    VehicleDetector vd;
    ros::spin();
    return 0;
}
