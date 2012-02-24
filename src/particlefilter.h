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


class Particle {
public:
    cv::Point2f pt;
    float rank;
    float acc_rank;

    Particle(int _x, int _y);

    void addXY(float _x, float _y);

    void calcRank(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints);

    void calcAccRank(vector<Particle>::iterator prev);

    bool operator<(const float b_acc_rank) const;

};


class ParticleFilter {
    int num_particles;
    cv::Ptr< vector<Particle> > particles;
    int mx, my;

    boost::mt19937 rng;
    boost::normal_distribution<float> norm_dist;
    boost::variate_generator< boost::mt19937&, boost::normal_distribution<float> > error_prop;

    Particle& searchByRank(const float rank);
public:

    ParticleFilter(const int _particles = 1024, const int _mx = 640, const int _my = 480, const float max_std_dev = 12.);

    void update(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints);

    void printParticles(cv::Mat& image);
};


#endif	/* PARTICLEFILTER_H */

