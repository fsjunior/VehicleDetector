#include "detector.h"
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>



using std::vector;


static const char WINDOW[] = "Vehicle Detector";
static const char REFERENCE_CAR[] = "images/carro.png";
//static const char REFERENCE_CAR[] = "images/carro2.png";
static const char SCENE_TOPIC[] = "/stereo/left/image_rect_mono";
//static const char SCENE_TOPIC[] = "/left/image_rect_mono";
static const float KEYPOINTS_FACTOR = 0.4;

static const std::string LOG_FILE = "data/log.dat";

enum {HOMOGRAPHY_PHASE, PARTICLE_PHASE, END };

/* Functions */
bool compKeyPoint(const cv::KeyPoint& a, const cv::KeyPoint& b)
{
    return a.response > b.response;
}

float sigmoid(const float x)
{
    return std::tanh(x);
}

void mouseCallBack(int event, int x, int y, int flags, void* param)
{
    DataLogger *dl = reinterpret_cast<DataLogger *>(param);
    
    if(cv::EVENT_LBUTTONDOWN || cv::EVENT_RBUTTONDOWN)
        switch(dl->status) {
            case HOMOGRAPHY_PHASE:

                dl->status = PARTICLE_PHASE;
                break;
            case PARTICLE_PHASE:

                dl->status = END;
                break;
        };
}


/* DataLogger */
DataLogger::DataLogger(const std::string &filename_)
:rng(static_cast<unsigned> (std::time(0))),
maxRand(0, 100),
genRand(rng, maxRand)
{
    dataLog.open(filename_.data());
    if(!dataLog.is_open())
        ROS_ERROR("Error opening logfile %s.", filename_.data());

    dataLog.precision(2);
    dataLog.setf(std::ios::fixed, std::ios::floatfield);
    //dataLog.setf(std::ios::showpoint);
    
}

void DataLogger::saveData()
{
    static int id = 1;
    //ID Homography_Computed Homographt_Detected Homography_X Homography_Y ParticleFilter_Detected ParticleFilter_X ParticleFilter_Y ParticleFilter_StdDev

    dataLog << id << " "

            << homographyComputed << " "
            << homographyDetected << " "
            << homographyCenter.x << " "
            << homographyCenter.y << " "

            << particleFilterDetected << " "
            << particleFilterCenter.x << " "
            << particleFilterCenter.y << " "
            << particleFilterStdDev << std::endl;

    
    id++;
}

DataLogger::~DataLogger()
{
    dataLog.close();
}

/* Vehicle Features */

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

    for(vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
        if(i->distance < min_dist)
            min_dist = i->distance;
        if(i->distance > max_dist)
            max_dist = i->distance;
        lmid_dist += i->distance;
    }

    lmid_dist /= (float) matches.size();

    if(mid_dist == 0.)
        mid_dist = lmid_dist;
    else
        mid_dist = mid_dist * 0.95 + lmid_dist * 0.05;
    ROS_DEBUG("mid_dist: %f; local_mid_dist %f", mid_dist, lmid_dist);
    ROS_DEBUG("Min/Max dist: %f %f", min_dist, max_dist);

    for(vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
        rank[i->queryIdx] = sigmoid(rank[i->queryIdx] + -(i->distance - mid_dist)*2.0);
        if(rank[i->queryIdx] >= 0.0)
            pairs.push_back(*i);
        else
            negpairs.push_back(*i);
    }
}

VehicleDetector::VehicleDetector()
: it_(nh_), frame_count(0), tt(0), dl(LOG_FILE)
{
    image_sub_ = it_.subscribe(SCENE_TOPIC, 10, &VehicleDetector::imageCb, this);

    detector = new cv::SurfFeatureDetector(400., 3, 4, true);
    extractor = new cv::SurfDescriptorExtractor(3, 4, false, true);

    if(detector.empty() || extractor.empty()) {
        ROS_ERROR("Error trying to initialize SURF Descipror.");
        ros::shutdown();
    }

    if(!loadReferenceCar()) {
        ROS_ERROR("Error opening reference car in: %s.", REFERENCE_CAR);
        ros::shutdown();
    }

    cv::namedWindow(WINDOW);
    cv::setMouseCallback(WINDOW, mouseCallBack, &dl);
}

VehicleDetector::~VehicleDetector()
{
    cv::destroyWindow(WINDOW);
}

void VehicleDetector::drawKeyPoints(cv::Mat& image, vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keyPoints, const cv::Scalar& color, int thickness)
{
    vector<cv::KeyPoint>::iterator k;

    for(vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
        k = keyPoints.begin() + i->trainIdx;
        cv::circle(image, k->pt, k->size * 1.2 / 9. * 2, color, thickness);
    }
}

cv::Point2f VehicleDetector::getVehicleCentralPoint(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &sceneKeyPoints)
{
    vector<cv::Point2f> src, dst;
    vector<cv::Point2f> srcpt, dstpt;



    for(vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
        src.push_back(referenceCar.keyPoints[i->queryIdx].pt);
        dst.push_back(sceneKeyPoints[i->trainIdx].pt);
    }

    cv::Mat h = cv::findHomography(src, dst, cv::LMEDS);

    srcpt.push_back(cv::Point2f(96., 42.));

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
    cv::Point2f homographyCenter(-1., -1.);


    try {
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());

        return;
    }

    detector->detect(cv_ptr->image, sceneKeypoints);

    if(sceneKeypoints.size() > 0) {
        extractor->compute(cv_ptr->image, sceneKeypoints, sceneDescriptors);

        referenceCar.findPairs(sceneDescriptors, pairs, negpairs);
        if(pairs.size() >= 4)
            cv::circle(cv_ptr->image, homographyCenter = getVehicleCentralPoint(pairs, sceneKeypoints), 2, cv::Scalar(0, 255, 255), 3);

        drawKeyPoints(cv_ptr->image, pairs, sceneKeypoints, cv::Scalar(0, 255, 0), 2);
        //drawKeyPoints(cv_ptr->image, negpairs, sceneKeypoints, cv::Scalar(0, 0, 255), 1);


        //cv::drawKeypoints(cv_ptr->image, sceneKeypoints, cv_ptr->image, cv::Scalar(255, 0, 0));

        pf.update(pairs, sceneKeypoints);

        pf.printParticles(cv_ptr->image);

    }

    //cv::drawKeypoints(referenceCarImage, referenceCar.keyPoints, referenceCarImage, cv::Scalar(255, 0, 0));
    //cv::imshow(WINDOW, referenceCarImage);

    

    frame_count++;

    tt = tt + (cv::getTickCount() - tl);

    if(dl.genRand() < 100) {
        dl.homographyComputed = pairs.size() >= 4;
        dl.homographyCenter = homographyCenter;


        dl.particleFilterCenter = pf.getPoint();
        //dl.particleFilterStatus = pf.getStatus();
        dl.particleFilterStdDev = pf.getStdDev();


        dl.status = HOMOGRAPHY_PHASE;
        cv::imshow(WINDOW, cv_ptr->image);
        while(dl.status != END && cv::waitKey(10) != 27);

        dl.saveData();
    }
    
}

bool VehicleDetector::loadReferenceCar()
{

    referenceCarImage = cv::imread(REFERENCE_CAR, CV_LOAD_IMAGE_GRAYSCALE);
    if(referenceCarImage.data) {
        detector->detect(referenceCarImage, referenceCar.keyPoints);
        ROS_INFO("Keypoints in object: %d; Selecting the best %d", (int) referenceCar.keyPoints.size(), (int) ((float) referenceCar.keyPoints.size() * KEYPOINTS_FACTOR));
        std::sort(referenceCar.keyPoints.begin(), referenceCar.keyPoints.end(), compKeyPoint);
        referenceCar.keyPoints.resize((int) ((float) referenceCar.keyPoints.size() * KEYPOINTS_FACTOR));
        referenceCar.rank.resize(referenceCar.keyPoints.size());

        extractor->compute(referenceCarImage, referenceCar.keyPoints, referenceCar.descriptors);

        cv::drawKeypoints(referenceCarImage, referenceCar.keyPoints, referenceCarImage, cv::Scalar(255, 0, 0));



        return true;
    }
    return false;
}

float VehicleDetector::getFPS()
{
    return frame_count / ((float) tt / (cv::getTickFrequency()));
}


