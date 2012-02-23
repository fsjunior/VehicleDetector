#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <boost/random.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <ctime>


using std::vector;


static const char WINDOW[] = "Vehicle Detector";
static const char REFERENCE_CAR[] = "images/carro.png";
static const float KEYPOINTS_FACTOR = 0.3;
static const unsigned int DYNAMIC_FEATURE_COUNT = 10;

unsigned long long frame_count = 0;
int64 tt = 0;



/* Functions */
bool compKeyPoint(const cv::KeyPoint& a, const cv::KeyPoint& b)
{
    return a.response > b.response;
}

float sigmoid(float x)
{
    return std::tanh(x);
}

/* Classes */

class Particle {
public:
    cv::Point2f pt;
    double rank;
    double acc_rank;

    Particle(int _x, int _y): pt(_x, _y), rank(0.), acc_rank(0.) {};

    void addXY(float _x, float _y)
    {
        pt.x += _x;
        pt.y += _y;
    }

    void calcRank(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints)
    {
        vector<cv::KeyPoint>::iterator k;
        float min_dist;

        k = keypoints.begin() + matches.begin()->trainIdx;

        min_dist = cv::norm(k->pt - pt);

        for (vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
            k = keypoints.begin() + i->trainIdx;

            float dist = cv::norm(k->pt - pt);
            min_dist = (dist < min_dist)? dist : min_dist;
        }

        rank = 1000. / min_dist;
        //ROS_INFO("%f %f", min_dist, rank);
    }

    void calcAccRank(vector<Particle>::iterator prev)
    {
        acc_rank = rank + prev->acc_rank;
        //ROS_INFO("%f", acc_rank);
    }
};

class ParticleFilter {
    int num_particles;
    cv::Ptr< vector<Particle> > particles;
    int mx, my;

    boost::mt19937 rng;
    boost::normal_distribution<float> norm_dist;
    boost::variate_generator< boost::mt19937&, boost::normal_distribution<float> > error_prop;


    Particle& searchByRank(int rank)
    {
        for(vector<Particle>::iterator i = particles->begin(); i != particles->end(); i++)
            if(rank < i->acc_rank)
                return *i;

                
    }

public:
    ParticleFilter(int _particles = 1000, int _mx = 640, int _my = 480, float max_std_dev = 10.): num_particles(_particles), mx(_mx), my(_my),
            rng(static_cast<unsigned> (std::time(0))),
            norm_dist(0., max_std_dev),
            error_prop(rng, norm_dist)
    {

        boost::uniform_int<> maxx(0, _mx);
        boost::uniform_int<> maxy(0, _my);

        boost::variate_generator< boost::mt19937&, boost::uniform_int<> > genx(rng, maxx);
        boost::variate_generator< boost::mt19937&, boost::uniform_int<> > geny(rng, maxy);

        particles = new vector<Particle>;

        for(int i = 0; i < _particles; i++)
            particles->push_back(Particle(genx(), geny()));
        
    };




    void update(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints)
    {
        cv::Ptr< vector<Particle> > new_particles;
        
        for(vector<Particle>::iterator i = particles->begin(); i != particles->end(); i++) {
            i->addXY(error_prop(), error_prop()); //sample_motion_model

            i->calcRank(matches, keypoints); //measurement_model

            i->calcAccRank(i - 1);         
        }

        //ROS_INFO("%f", (particles->end() - 1)->acc_rank);
        boost::uniform_real<double> max_rank(0., (particles->end() - 1)->acc_rank);
        boost::variate_generator< boost::mt19937&, boost::uniform_real<double> > rank_rand(rng, max_rank);


        new_particles = new vector<Particle>;

        for(int i = 0; i < num_particles; i++) {
            int rand = rank_rand();
            new_particles->push_back(searchByRank(rand));
        }

        //particles.delete_obj();
        
        particles = new_particles;

    }

    void printParticles (cv::Mat& image)
    {
        for(vector<Particle>::iterator i = particles->begin(); i != particles->end(); i++)
            cv::circle(image, i->pt, 2, cv::Scalar(255, 0, 0), 1);
           
    }


};


class VehicleFeatures {
private:
    float mid_dist;
public:
    vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    vector< float > rank;
    cv::FlannBasedMatcher matcher;


    VehicleFeatures() : mid_dist(0.)//, matcher(new cv::flann::KDTreeIndexParams(4))
    {
        frame_count = 0;
        tt = 0;
    };

    void findPairs(cv::Mat& sceneDescriptors, vector<cv::DMatch>& pairs, vector<cv::DMatch>& negpairs)
    {

        //cv::BruteForceMatcher< cv::L2<float> > matcher;
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
            //ROS_DEBUG("RANK: [%d]: %f", i->queryIdx, rank[i->queryIdx]);
            if (rank[i->queryIdx] >= 0.0)
                pairs.push_back(*i);
            else
                negpairs.push_back(*i);
        }
    }
};

class VehicleDetector {
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;

    VehicleFeatures referenceCar;

    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;

    ParticleFilter pf;

public:

    

    VehicleDetector()
    : it_(nh_)
    {
        image_sub_ = it_.subscribe("/stereo/left/image_rect_mono", 10, &VehicleDetector::imageCb, this);

        detector = new cv::SurfFeatureDetector(400., 3, 4, true);
        //detector = new cv::GoodFeaturesToTrackDetector();
        //detector = new cv::StarFeatureDetector();

        extractor = new cv::SurfDescriptorExtractor(3, 4, false, true);
        //extractor = new cv::BriefDescriptorExtractor(64);

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

    ~VehicleDetector()
    {

        cv::destroyWindow(WINDOW);

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


        drawKeyPoints(cv_ptr->image, pairs, sceneKeypoints, cv::Scalar(0, 255, 0), 2);
        drawKeyPoints(cv_ptr->image, negpairs, sceneKeypoints, cv::Scalar(0, 0, 255), 1);

        pf.update(pairs, sceneKeypoints);
        
        pf.printParticles(cv_ptr->image);
        
        cv::imshow(WINDOW, cv_ptr->image);

        frame_count++;

        tt = tt + (cv::getTickCount() - tl);

        cv::waitKey(3);
    }

    bool loadReferenceCar()
    {
        cv::Mat referenceCarImage;
        referenceCarImage = cv::imread(REFERENCE_CAR, CV_LOAD_IMAGE_GRAYSCALE);
        if (referenceCarImage.data) {
            detector->detect(referenceCarImage, referenceCar.keyPoints);

            ROS_INFO("Keypoints in object: %d; Selecting the best %d", referenceCar.keyPoints.size(), (int) ((float) referenceCar.keyPoints.size() * KEYPOINTS_FACTOR));
            std::sort(referenceCar.keyPoints.begin(), referenceCar.keyPoints.end(), compKeyPoint);
            referenceCar.keyPoints.resize((int) ((float) referenceCar.keyPoints.size() * KEYPOINTS_FACTOR));
            referenceCar.rank.resize(referenceCar.keyPoints.size());

            extractor->compute(referenceCarImage, referenceCar.keyPoints, referenceCar.descriptors);

            return true;
        }
        return false;
    }





};

int main(int argc, char** argv)
{
    double time_elapsed;
    
    ros::init(argc, argv, "vehicle_detector");

    VehicleDetector vd;
    ros::spin();

    

    time_elapsed = (double)tt/(cv::getTickFrequency());
    std::cout << "Frames por segundo: " << (double)frame_count/time_elapsed << std::endl;
    return 0;
}
