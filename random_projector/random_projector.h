#ifndef RANDOM_PROJ_H
#define RANDOM_PROJ_H

#include <vector>
#include <Eigen/Core>
#include "split_loss.h"

class RandomProjector
{
public:
    std::vector<double> bestSplits;
    std::vector<double> bestLosses;
    std::vector<std::vector<double>> projectionVectors;
    RandomProjector(int nProjections, int nSplits, int nFeatures, int nClasses, int nSamples,
                     double* Anlist, int AnlistSize, bool random, double *X, double *y);
    void runMultithread();
    void runSinglethread();
    void runGPU();

private:
    // const data
    const int _nProjections;
    const int _nSplits;
    const int _nFeatures;
    const int _nClasses;
    const int _nSamples;
    const double *_Anlist;
    const int _AnlistSize;
    const bool _random;
    Eigen::MatrixXd _X;
    const double *_y;

    // thread data
    std::vector<Eigen::VectorXd> _projectedX;
    std::vector<std::vector<double>> _splits;

    // helper functions
    Eigen::VectorXd AvecEnvelop(int iteration, double *Anlist, int AnlistSize, bool random);
    std::vector<double> generateSplit(Eigen::VectorXd projectedX, int nSplits);
    double calculateSELoss(Eigen::VectorXd projectedX, double split);
    double calculateWELoss(Eigen::VectorXd projectedX, double split);
    
    // thread functions
    void batchedCalcWELoss(int startIdx, int endIdx);
    void batchedCalcSELoss(int startIdx, int endIdx);

};

#endif