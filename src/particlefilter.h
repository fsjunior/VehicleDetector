/* 
 * File:   particlefilter.h
 * Author: linuxer
 *
 * Created on February 24, 2012, 12:22 AM
 */

#ifndef PARTICLEFILTER_H
#define	PARTICLEFILTER_H

#include <opencv2/features2d/features2d.hpp>
#include <boost/random.hpp>
#include <ctime>


using std::vector;

namespace pf {


    enum {DETECTED, CAUTION, NOT_DETECTED};
   

    class Particle {
    public:
        cv::Point2f pt;
        float rank;
        float acc_rank;

        Particle(const float _x = 0., const float _y = 0.);
        Particle(const Particle &p);

        void stepSampleMotionModel(const float _x, const float _y);

        void stepMeasurementModel(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints, vector<Particle>::iterator prev);

        bool operator<(const float b_acc_rank) const;

        Particle operator +(const Particle& b) const;
        Particle operator *(const Particle& b) const;
    };
    

    class ParticleFilter {
        int num_particles;
        cv::Ptr< vector<Particle> > particles;
        int mx, my;
        float max_std_dev;

        cv::Point2f mean;
        float stddev;


        boost::mt19937 rng;
        boost::normal_distribution<float> norm_dist;
        boost::variate_generator< boost::mt19937&, boost::normal_distribution<float> > error_prop;

        Particle& searchByRank(const float rank);

        
    public:

        ParticleFilter(const int _particles = 1024, const int _mx = 640, const int _my = 480, const float _max_std_dev = 10.);

        void update(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints);

        int getStatus();
        float getStdDev();

        cv::Point2f& getPoint();

        void printParticles(cv::Mat& image);
    };

};

#endif	/* PARTICLEFILTER_H */

