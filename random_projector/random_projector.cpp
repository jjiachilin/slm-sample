#include "random_projector.h"
#include "float.h"
#include "stdio.h"
#include <vector>
#include <future>
#include <boost/asio.hpp>
#include <Eigen/Core>
#include <iostream>
#include "split_loss.h"

RandomProjector::RandomProjector(int nProjections, int nSplits, int nFeatures, int nClasses, int nSamples,
                                 double *Anlist, int AnlistSize, bool random, double *X, double *y)
    : _nProjections(nProjections), _nSplits(nSplits), _nFeatures(nFeatures), _nClasses(nClasses),
      _nSamples(nSamples), _Anlist(Anlist), _AnlistSize(AnlistSize), _random(random), _y(y)
{
    _X = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(X, nSamples, nFeatures);
    // init lists
    for (int i = 0; i < nProjections; ++i)
    {
        bestSplits.push_back(-1);
        bestLosses.push_back(DBL_MAX);
        Eigen::VectorXd Avec = AvecEnvelop(i, Anlist, AnlistSize, random);
        std::vector<double> stdAvec(Avec.data(), Avec.data() + Avec.rows() * Avec.cols());
        projectionVectors.push_back(stdAvec);
        _projectedX.push_back(_X * Avec);
        _splits.push_back(generateSplit(_projectedX.back(), nSplits));
    }
}

Eigen::VectorXd RandomProjector::AvecEnvelop(int iteration, double *Anlist, int AnlistSize, bool random)
{
    Eigen::VectorXd Avec(AnlistSize);
    if (random)
    {
        for (int i = 0; i < AnlistSize; ++i)
        {
            Avec(i) = -Anlist[i] + rand() % (int)((Anlist[i] + 1) - (-Anlist[i]));
        }
        bool allZero = true;
        for (int i = 0; i < AnlistSize; ++i)
            if (Avec(i) != 0)
                allZero = false;
        if (allZero)
            Avec(rand() % AnlistSize) = 1;
    }
    else
    {
        int i = 0;
        while (i != 0 && i < AnlistSize)
        {
            Avec(i) = -Anlist[i] + (iteration % (int)(2 * Anlist[i] + 1));
            iteration = iteration / (2 * Anlist[i] + 1);
            ++i;
        }
    }
    return Avec;
}

std::vector<double> RandomProjector::generateSplit(Eigen::VectorXd X, int nSplits)
{
    double Xmin = X.minCoeff();
    double Xmax = X.maxCoeff();
    double linspace = (Xmax - Xmin) / (nSplits + 2);
    std::vector<double> split;
    for (int i = 0; i < nSplits; ++i)
    {
        split.push_back(Xmin + linspace);
        Xmin += linspace;
    }
    return split;
}

// returns squared error loss
double RandomProjector::calculateSELoss(Eigen::VectorXd projectedX, double split)
{
    int leftySize = 0;
    int rightySize = 0;
    double leftySum = 0;
    double rightySum = 0;

    for (int j = 0; j < _nSamples; ++j)
    {
        if (projectedX[j] <= split)
        {
            ++leftySize;
            leftySum += _y[j];
        }
        else
        {
            ++rightySize;
            rightySum += _y[j];
        }
    }

    double leftyMean = leftySum / leftySize;
    double rightyMean = rightySum / rightySize;
    double leftyH = 0;
    double rightyH = 0;

    for (int j = 0; j < _nSamples; ++j)
    {
        if (projectedX[j] <= split)
        {
            leftyH += (_y[j] - leftyMean) * (_y[j] - leftyMean);
        }
        else
        {
            rightyH += (_y[j] - rightyMean) * (_y[j] - rightyMean);
        }
    }

    return (leftySize / (double)_nSamples) * leftyH + (rightySize / (double)_nSamples) * rightyH;
}

// returns weighted entropy loss
double RandomProjector::calculateWELoss(Eigen::VectorXd projectedX, double split)
{
    double leftyProbs[100];
    double rightyProbs[100];

    int leftySize = 0;
    int rightySize = 0;
    for (int i = 0; i < _nClasses; ++i)
    {
        leftyProbs[i] = 0;
        rightyProbs[i] = 0;
    }

    for (int j = 0; j < _nSamples; ++j)
    {
        int maxIdx = 0;
        if (projectedX[j] <= split)
        {
            ++leftySize;
            for (int k = 1; k < _nClasses; ++k)
            {
                if (_y[j * _nClasses + k] > _y[j * _nClasses + maxIdx])
                {
                    maxIdx = k;
                }
            }
            ++leftyProbs[maxIdx];
        }
        else
        {
            ++rightySize;
            for (int k = 1; k < _nClasses; ++k)
            {
                if (_y[j * _nClasses + k] > _y[j * _nClasses + maxIdx])
                {
                    maxIdx = k;
                }
            }
            ++rightyProbs[maxIdx];
        }
    }

    for (int j = 0; j < _nClasses; ++j)
    {
        leftyProbs[j] /= (double)leftySize;
        rightyProbs[j] /= (double)rightySize;
    }

    double rightyE = 0;
    double leftyE = 0;

    for (int j = 0; j < _nClasses; ++j)
    {
        if (rightyProbs[j] > 0)
            rightyE -= rightyProbs[j] * log(rightyProbs[j]);
        if (leftyProbs[j] > 0)
            leftyE -= leftyProbs[j] * log(leftyProbs[j]);
    }

    rightyE /= log(_nClasses);
    leftyE /= log(_nClasses);

    return (leftySize / (double)_nSamples) * leftyE + (rightySize / (double)_nSamples) * rightyE;
}

// each thread should receive start and end index of nprojections*nsplits to calculate loss on
void RandomProjector::batchedCalcSELoss(int startIdx, int endIdx)
{
    int projectionIdx;
    int splitIdx;
    double loss;
    for (int i = startIdx; i < endIdx; ++i)
    {
        projectionIdx = (int)(i / _nSplits);
        splitIdx = i % _nSplits;
        loss = calculateSELoss(_projectedX[projectionIdx], _splits[projectionIdx][splitIdx]);
        if (loss < bestLosses[projectionIdx])
        {
            bestLosses[projectionIdx] = loss;
            bestSplits[projectionIdx] = _splits[projectionIdx][splitIdx];
        }
    }
}

void RandomProjector::batchedCalcWELoss(int startIdx, int endIdx)
{
    int projectionIdx;
    int splitIdx;
    double loss;
    for (int i = startIdx; i < endIdx; ++i)
    {
        projectionIdx = (int)(i / _nSplits);
        splitIdx = i % _nSplits;
        loss = calculateWELoss(_projectedX[projectionIdx], _splits[projectionIdx][splitIdx]);
        if (loss < bestLosses[projectionIdx])
        {
            bestLosses[projectionIdx] = loss;
            bestSplits[projectionIdx] = _splits[projectionIdx][splitIdx];
        }
    }
}

void RandomProjector::runMultithread()
{
    // init thread pool
    const int nThreads = std::min(_nProjections * _nSplits, (int)std::thread::hardware_concurrency());
    boost::asio::thread_pool pool(nThreads);

    // partition nProjections x nSplits calculations to nThreads chunks for multithreaded processing
    int chunkSize = (int)(_nProjections * _nSplits) / nThreads;
    int lastChunkSize = (_nProjections * _nSplits) - (chunkSize * nThreads) + chunkSize;

    // launch nthreads for each a batch of loss calculation tasks
    int i = 0;
    while (i < _nProjections * _nSplits)
    {
        if (_nProjections * _nSplits - i < 2 * chunkSize)
            chunkSize = lastChunkSize;

        if (_nClasses > 1)
            boost::asio::post(pool, std::bind(&RandomProjector::batchedCalcWELoss, this, i, i + chunkSize));
        else
            boost::asio::post(pool, std::bind(&RandomProjector::batchedCalcSELoss, this, i, i + chunkSize));

        i += chunkSize;
    }
    pool.join();
}

void RandomProjector::runSinglethread()
{
    double loss;
    for (int i = 0; i < _nProjections; ++i)
    {
        for (double split : _splits[i])
        {
            if (_nClasses > 1)
            {
                loss = calculateWELoss(_projectedX[i], split);
                if (loss < bestLosses[i])
                {
                    bestLosses[i] = loss;
                    bestSplits[i] = split;
                }
            }
            else
            {
                loss = calculateSELoss(_projectedX[i], split);
                if (loss < bestLosses[i])
                {
                    bestLosses[i] = loss;
                    bestSplits[i] = split;
                }
            }
        }
    }
}

void RandomProjector::runGPU()
{
    // pack data into c buffers
    double *AVectorsBuf = (double *)malloc(sizeof(double) * _nProjections * _nFeatures);
    double *projectedXBuf = (double *)malloc(sizeof(double) * _nProjections * _nSamples);
    double *splitsBuf = (double *)malloc(sizeof(double) * _nProjections * _nSplits);
    for (int i = 0; i < _nProjections; ++i)
    {
        for (int j = 0; j < _nFeatures; ++j)
            AVectorsBuf[i * _nFeatures + j] = projectionVectors[i].data()[j];
        for (int j = 0; j < _nSamples; ++j)
            projectedXBuf[i * _nSamples + j] = _projectedX[i].data()[j];
        for (int j = 0; j < _nSplits; ++j)
            splitsBuf[i * _nSplits + j] = _splits[i][j];
    }
    // contains calculation results
    double *splitLossesBuf = (double *)malloc(sizeof(double) * _nProjections * _nSplits);
    if (_nClasses > 1)
        gpuCalculateSplitWELosses(_nProjections, _nSplits, _nFeatures, _nSamples, _nClasses,
                                  projectedXBuf, splitsBuf, _y, splitLossesBuf);
    else
        gpuCalculateSplitSELosses(_nProjections, _nSplits, _nFeatures, _nSamples,
                                  projectedXBuf, splitsBuf, _y, splitLossesBuf);
    // retrieve minimums
    for (int i = 0; i < _nProjections; ++i)
    {
        for (int j = 0; j < _nSplits; ++j)
        {
            if (splitLossesBuf[i * _nSplits + j] < bestLosses[i])
            {
                bestLosses[i] = splitLossesBuf[i * _nSplits + j];
                bestSplits[i] = _splits[i][j];
            }
        }
    }
    free(AVectorsBuf);
    free(projectedXBuf);
    free(splitsBuf);
    free(splitLossesBuf);
}

/*
int main()
{
    srand(1);
    unsigned int nProjections = 1000;
    unsigned int nSplits = 64;
    unsigned int nClasses = 2;
    unsigned int nFeatures = 4;	   // assume we've already chosen a feature to split on
    unsigned int nSamples = 1000; // this is the dimension of training data
    unsigned int AnlistSize = nFeatures;
    // generate random X vector and mock Anlist vector
    // X.shape = nSamples, nFeatures
    // Anlist.shape = nFeatures, 1
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(nFeatures, nSamples);
    Eigen::MatrixXd y = Eigen::MatrixXd::Zero(nClasses, nSamples);
    for (unsigned int i = 0; i < nSamples; ++i)
    {
        y(rand() % nClasses, i) = 1;
    }
    Eigen::VectorXd Anlist(AnlistSize);
    Anlist << 1, 2, 3, 4;
    RandomProjector rp = RandomProjector(nProjections, nSplits, nFeatures, nClasses, nSamples,
                                           Anlist.data(), AnlistSize, true, X.data(), y.data());
    rp.runGPU();
    for (int i = 0; i < nProjections; ++i)
    {
        std::cout << rp.bestLosses[i] << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < nProjections; ++i)
    {
        std::cout << rp.bestSplits[i] << " ";
    }
    std::cout << "\n";
}*/
