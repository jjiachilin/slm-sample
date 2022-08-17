#ifndef APSO_H
#define APSO_H

#include <Eigen/Core>
#include <vector>
#include <random>
#include <mutex>

std::mutex mtx;

class APSO
{
public:
    std::vector<double> bestSplits;
    std::vector<double> bestLosses;
    std::vector<std::vector<double>> bestParticles;
    int _globalBestParticleIdx;
    // constructor
    APSO(int nParticles, int nSplits, int nFeatures, int nClasses, int nSamples,
         double w, int maxIter, int minAn, double c1, double c2, double percent,
         double stdMin, double stdMax, double *X, double *y);
    // update function for max iterations
    void run();

private:
    // for dimensioning data
    const int _nParticles;
    const int _nSplits;
    const int _nFeatures;
    const int _nClasses;
    const int _nSamples;

    // pso hyperparameters
    const int _maxIter;
    const int _minAn;
    const double _vMax;

    // adaptive hyperparameters
    double _w;
    double _c1;
    double _c2;
    int _state;
    double _stdMin;
    double _stdMax;

    // const data
    Eigen::MatrixXd _X;
    const double *_y;

    // data that is updated
    double _globalBestLoss;
    Eigen::VectorXd _globalBestParticle;
    
    std::vector<Eigen::VectorXd> _particles;
    std::vector<Eigen::VectorXd> _velocities;

    std::default_random_engine _generator;
    // for multithreading
    // int updateVelocitiesParticles(int particleIdx);
    // int calculateBestLosses(int particleIdx);
    std::pair<double, double> threadCalculateSplitWELoss(Eigen::VectorXd projectedX, std::vector<double> splits);
    std::pair<double, double> threadCalculateSplitSELoss(Eigen::VectorXd projectedX, std::vector<double> splits);

    // helper functions
    std::vector<double> generateSplit(Eigen::MatrixXd X);
    void getProjectionsSplits(double *projectedXBuf, double *splitsBuf);
    void cpuCalculateSplitWELosses(double *projectedX, double *splits, double *splitLosses, int nParticles);
    void cpuCalculateSplitSELosses(double *projectedX, double *splits, double *splitLosses, int nParticles);
    std::vector<double> euclidDist(Eigen::VectorXd u, std::vector<Eigen::VectorXd> v);
    std::vector<double> calDistance();
    double calF(double dMax, double dMin, double dBest);
    void identifyState(double f);
    double f1(double f);
    double f2(double f);
    double f3(double f);
    double f4(double f);
    void updateC(int iter);
    void doEliteLearning(int iter);
    void updateInertia(double f);
    void updateParticles();
    Eigen::VectorXd clipVec(Eigen::VectorXd particle, double least, double greatest);
    Eigen::VectorXd roundVec(Eigen::VectorXd particle);
};

#endif
