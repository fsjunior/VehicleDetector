#include <boost/lexical_cast.hpp>

#include "particlefilter.h"
#include "numeric"
#include <stdio.h>

using namespace pf;

/* Particle */
Particle::Particle(const float _x, const float _y) : pt(_x, _y), rank(0.), acc_rank(0.)
{
};

Particle::Particle(const Particle &p)
{
    pt = p.pt;
    rank = p.rank;
    acc_rank = p.acc_rank;
}

void Particle::stepSampleMotionModel(const float _x, const float _y)
{
    pt.x += _x;
    pt.y += _y;
}

void Particle::stepMeasurementModel(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints, vector<Particle>::iterator prev)
{
    vector<cv::KeyPoint>::iterator k;
    float min_dist;

    k = keypoints.begin() + matches.begin()->trainIdx;

    min_dist = cv::norm(k->pt - pt);

    for(vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
        k = keypoints.begin() + i->trainIdx;

        float dist = cv::norm(k->pt - pt);
        min_dist = (dist < min_dist) ? dist : min_dist;
    }

    rank = 1000. / min_dist;
    acc_rank = rank + prev->acc_rank;
}

bool Particle::operator<(const float b_acc_rank) const
{
    return acc_rank < b_acc_rank;
}

Particle Particle::operator +(const Particle& b) const
{
    return Particle(pt.x + b.pt.x, pt.y + b.pt.y);
}

Particle Particle::operator *(const Particle& b) const
{
    return Particle(pt.x * b.pt.x, pt.y * b.pt.y);
}

/*ParticleFilter */
Particle& ParticleFilter::searchByRank(const float rank)
{
    return *std::lower_bound(particles->begin(), particles->end(), rank);
}

ParticleFilter::ParticleFilter(const int _particles, const int _mx, const int _my, const float _max_std_dev)
: num_particles(_particles), mx(_mx), my(_my), max_std_dev(_max_std_dev),
rng(static_cast<unsigned> (std::time(0))),
norm_dist(0., _max_std_dev),
error_prop(rng, norm_dist)
{
    boost::uniform_int<> maxx(0, _mx);
    boost::uniform_int<> maxy(0, _my);

    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > genx(rng, maxx);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > geny(rng, maxy);

    particles = new vector<Particle>;

    for(int i = 0; i < _particles; i++)
        particles->push_back(Particle(genx(), geny()));
}

void ParticleFilter::update(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints)
{
    cv::Ptr< vector<Particle> > new_particles;

    for(vector<Particle>::iterator i = particles->begin(); i != particles->end(); i++) {
        i->stepSampleMotionModel(error_prop(), error_prop()); //sample_motion_model
        i->stepMeasurementModel(matches, keypoints, i - 1); //measurement_model
    }

    boost::uniform_real<float> max_rank(0., (particles->end() - 1)->acc_rank);
    boost::variate_generator< boost::mt19937&, boost::uniform_real<float> > rank_rand(rng, max_rank);


    new_particles = new vector<Particle>;

    Particle sum(0., 0.);

    for(int i = 0; i < num_particles; i++) {
        float rand = rank_rand();
        Particle *p = &searchByRank(rand);

        new_particles->push_back(*p);
        sum = sum + *p;
    }

    /* Calcula média e desvio padrão */
    //Particle sum = std::accumulate(new_particles->begin(), new_particles->end(), Particle());
    mean.x = sum.pt.x / (float) num_particles;
    mean.y = sum.pt.y / (float) num_particles;

    Particle sqsum = std::inner_product(new_particles->begin(), new_particles->end(), new_particles->begin(), Particle());

    stddev = (std::sqrt(sqsum.pt.x / (float) num_particles - mean.x * mean.x) + std::sqrt(sqsum.pt.y / (float) num_particles - mean.y * mean.y)) / 2.;

    particles = new_particles;
}

int ParticleFilter::getStatus()
{
    if(stddev < max_std_dev * 1.5)
        return DETECTED;
    else if(stddev > max_std_dev * 1.5 && stddev < max_std_dev * 2.)
        return CAUTION;
    else
        return NOT_DETECTED;
}

float ParticleFilter::getStdDev()
{
    return stddev;
}

cv::Point2f& ParticleFilter::getPoint()
{
    return mean;
}

void ParticleFilter::printParticles(cv::Mat& image)
{
    for(vector<Particle>::iterator i = particles->begin(); i != particles->end(); i++)
        cv::circle(image, i->pt, 1, cv::Scalar(255, 0, 0), 1);

    cv::Scalar color;

    switch(getStatus()) {
        case DETECTED: color = cv::Scalar(0, 255, 0);
            break;
        case CAUTION: color = cv::Scalar(0, 255, 255);
            break;
        case NOT_DETECTED: color = cv::Scalar(0, 0, 255);
            break;
    }

    cv::circle(image, mean, 2, color, 2);
    cv::circle(image, mean, stddev, color, 1);

    std::ostringstream s;
    s << stddev;
    cv::Point2f p(mean);
    p.x += stddev;

    cv::putText(image, s.str(), p, cv::FONT_HERSHEY_PLAIN, 1, color, 1, 8, false);
}

