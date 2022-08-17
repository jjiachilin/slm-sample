#ifndef PSO_H
#define PSO_H

#include <Eigen/Core>
#include <vector>
#include <mutex>

std::mutex mtx;

class ParticleSwarmOptimizer
{
public:
    std::vector<double> bestSplits;
    std::vector<double> bestLosses;
    std::vector<std::vector<double>> bestParticles;
    // constructor
    ParticleSwarmOptimizer(int nParticles, int nSplits, int nFeatures, int nClasses, int nSamples,
                           double w, double c1, double c2, double r1, double r2, int maxIter, int minAn,
                           double *X, double *y);
    // update function for one iteration
    void updateMulti();
    void update();

private:
    // for dimensioning data
    const int _nParticles;
    const int _nSplits;
    const int _nFeatures;
    const int _nClasses;
    const int _nSamples;

    // pso hyperparameters
    const double _w;
    const double _c1;
    const double _c2;
    const double _r1;
    const double _r2;
    const int _maxIter;
    const int _minAn;

    // const data
    Eigen::MatrixXd _X;
    const double *_y;

    // data that is updated
    double _globalBestLoss;
    Eigen::VectorXd _globalBestParticle;
    std::vector<Eigen::VectorXd> _particles;
    std::vector<Eigen::VectorXd> _velocities;

    // for multithreading
    int updateVelocitiesParticles(int particleIdx);
    int calculateBestLosses(int particleIdx);
    std::pair<double, double> threadCalculateSplitWELoss(Eigen::VectorXd projectedX, int particleIdx, std::vector<double> splits);
    std::pair<double, double> threadCalculateSplitSELoss(Eigen::VectorXd projectedX, int particleIdx, std::vector<double> splits);

    // helper functions
    std::vector<double> generateSplit(Eigen::MatrixXd X);
    void getProjectionsSplits(double *projectedXBuf, double *splitsBuf);
    void cpuCalculateSplitWELosses(double *projectedX, double *splits, double *splitLosses);
    void cpuCalculateSplitSELosses(double *projectedX, double *splits, double *splitLosses);
};

#endif
