#include "particlefilter.h"

Particle::Particle(int _x, int _y) : pt(_x, _y), rank(0.), acc_rank(0.)
{
};

void Particle::addXY(float _x, float _y)
{
    pt.x += _x;
    pt.y += _y;
}

void Particle::calcRank(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints)
{
    vector<cv::KeyPoint>::iterator k;
    float min_dist;

    k = keypoints.begin() + matches.begin()->trainIdx;

    min_dist = cv::norm(k->pt - pt);

    for (vector<cv::DMatch>::iterator i = matches.begin(); i != matches.end(); i++) {
        k = keypoints.begin() + i->trainIdx;

        float dist = cv::norm(k->pt - pt);
        min_dist = (dist < min_dist) ? dist : min_dist;
    }

    rank = 1000. / min_dist;
    //ROS_INFO("%f %f", min_dist, rank);
}

void Particle::calcAccRank(vector<Particle>::iterator prev)
{
    acc_rank = rank + prev->acc_rank;
    //ROS_INFO("%f", acc_rank);
}

bool Particle::operator<(const float b_acc_rank) const
{
    return acc_rank < b_acc_rank;
}

Particle& ParticleFilter::searchByRank(const float rank)
{
    return *(std::lower_bound(particles->begin(), particles->end(), rank));
}

ParticleFilter::ParticleFilter(const int _particles, const int _mx, const int _my, const float max_std_dev) : num_particles(_particles), mx(_mx), my(_my),
rng(static_cast<unsigned> (std::time(0))),
norm_dist(0., max_std_dev),
error_prop(rng, norm_dist)
{

    boost::uniform_int<> maxx(0, _mx);
    boost::uniform_int<> maxy(0, _my);


    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > genx(rng, maxx);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > geny(rng, maxy);

    particles = new vector<Particle>;

    for (int i = 0; i < _particles; i++)
        particles->push_back(Particle(genx(), geny()));

};

void ParticleFilter::update(vector<cv::DMatch>& matches, vector<cv::KeyPoint> &keypoints)
{
    cv::Ptr< vector<Particle> > new_particles;

    for (vector<Particle>::iterator i = particles->begin(); i != particles->end(); i++) {
        i->addXY(error_prop(), error_prop()); //sample_motion_model

        i->calcRank(matches, keypoints); //measurement_model

        i->calcAccRank(i - 1);
    }

    //ROS_INFO("%f", (particles->end() - 1)->acc_rank);
    boost::uniform_real<float> max_rank(0., (particles->end() - 1)->acc_rank);
    boost::variate_generator< boost::mt19937&, boost::uniform_real<float> > rank_rand(rng, max_rank);


    new_particles = new vector<Particle>;

    for (int i = 0; i < num_particles; i++) {
        float rand = rank_rand();
        new_particles->push_back(searchByRank(rand));
    }


    particles = new_particles;
}

void ParticleFilter::printParticles(cv::Mat& image)
{
    for (vector<Particle>::iterator i = particles->begin(); i != particles->end(); i++)
        cv::circle(image, i->pt, 1, cv::Scalar(255, 0, 0), 1);
}

