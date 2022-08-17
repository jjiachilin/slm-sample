#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/asio.hpp>
#include <vector>
#include <math.h>
#include <float.h>
#include <iostream>
#include <thread>
#include <chrono>
#include "pso.h"
#include "../random_projector/split_loss.h"

void displayBuffer(double *head, int d1, int d2)
{
	for (int i = 0; i < d1; ++i)
	{
		for (int j = i * d2; j < (i + 1) * d2; ++j)
		{
			std::cout << head[j] << " ";
		}
		std::cout << std::endl;
	}
}

ParticleSwarmOptimizer::ParticleSwarmOptimizer(int nParticles, int nSplits, int nFeatures, int nClasses, int nSamples,
											   double w, double c1, double c2, double r1, double r2, int maxIter, int minAn,
											   double *X, double *y)
	: _nParticles(nParticles), _nSplits(nSplits), _nFeatures(nFeatures), _nClasses(nClasses), _nSamples(nSamples), _w(w), _c1(c1),
	  _c2(c2), _r1(r1), _r2(r2), _maxIter(maxIter), _minAn(minAn), _y(y)
{
	_X = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(X, nSamples, nFeatures);
	// init random particles, velocities, splits, losses
	for (int i = 0; i < nParticles; ++i)
	{
		Eigen::VectorXd Avec(nFeatures);
		for (int j = 0; j < nFeatures; ++j)
			Avec(j) = -minAn + rand() % (int)((minAn + 1) - (-minAn));

		bool allZero = true;
		for (int j = 0; j < nFeatures; ++j)
		{
			if (Avec(j) != 0)
			{
				allZero = false;
				break;
			}
		}
		if (allZero)
			Avec(rand() % nFeatures) = 1;

		_particles.push_back(Avec);
		// bestParticles Avec needs to be stl vector to map to cython easily
		std::vector<double> stdAvec(Avec.data(), Avec.data() + Avec.rows() * Avec.cols());
		bestParticles.push_back(stdAvec);
		bestLosses.push_back(DBL_MAX);
		bestSplits.push_back(0);
		_velocities.push_back(Eigen::VectorXd::Random(nFeatures));
	}
	_globalBestLoss = DBL_MAX;
	// init splits and bestLosses
	double *projectedXBuf = (double *)malloc(sizeof(double) * nParticles * nSamples);
	double *splitsBuf = (double *)malloc(sizeof(double) * nParticles * nSplits);
	double *splitLossesBuf = (double *)malloc(sizeof(double) * nParticles * nSplits);
	getProjectionsSplits(projectedXBuf, splitsBuf);

	if (_nClasses > 1) // classification
	{
		// gpuCalculateSplitWELosses(_nParticles, _nSplits, _nFeatures, _nSamples, _nClasses,
		// 					projectedXBuf, splitsBuf, _y, splitLossesBuf);
		cpuCalculateSplitWELosses(projectedXBuf, splitsBuf, splitLossesBuf);
	}
	else // regression
	{
		// gpuCalculateSplitSELosses(_nParticles, _nSplits, _nFeatures, _nSamples,
		// 					projectedXBuf, splitsBuf, _y, splitLossesBuf);
		cpuCalculateSplitSELosses(projectedXBuf, splitsBuf, splitLossesBuf);
	}

	// displayBuffer(minSplitLossesBuf, nParticles, 1);
	// displayBuffer(bestSplitsBuf, nParticles, 1);
	for (int i = 0; i < nParticles; ++i)
	{
		for (int j = i * nSplits; j < (i + 1) * nSplits; ++j)
		{
			if (splitLossesBuf[j] < bestLosses[i])
			{
				bestLosses[i] = splitLossesBuf[j];
				bestSplits[i] = splitsBuf[j];
				std::vector<double> stdParticle(_particles[i].data(), _particles[i].data() + _particles[i].rows() * _particles[i].cols());
				bestParticles[i] = stdParticle;
				if (splitLossesBuf[j] < _globalBestLoss)
				{
					_globalBestLoss = splitLossesBuf[j];
					_globalBestParticle = _particles[i];
				}
			}
		}
	}

	free(projectedXBuf);
	free(splitsBuf);
	free(splitLossesBuf);
}

void ParticleSwarmOptimizer::cpuCalculateSplitSELosses(double *projectedX, double *splits, double *splitLosses)
{
	for (int i = 0; i < _nParticles * _nSplits; ++i)
	{
		int projectedXOffset = (int)(i / _nSplits) * _nSamples;
		int leftySize = 0;
		int rightySize = 0;
		double leftySum = 0;
		double rightySum = 0;

		for (int j = 0; j < _nSamples; ++j)
		{
			if (projectedX[projectedXOffset + j] <= splits[i])
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
			if (projectedX[projectedXOffset + j] <= splits[i])
			{
				leftyH += (_y[j] - leftyMean) * (_y[j] - leftyMean);
			}
			else
			{
				rightyH += (_y[j] - rightyMean) * (_y[j] - rightyMean);
			}
		}
		splitLosses[i] = (leftySize / (double)_nSamples) * leftyH + (rightySize / (double)_nSamples) * rightyH;
	}
}

void ParticleSwarmOptimizer::cpuCalculateSplitWELosses(double *projectedX, double *splits, double *splitLosses)
{
	double leftyProbs[100]; // hopefully we won't have to predict more than 100 classes in the near future
	double rightyProbs[100];

	for (int i = 0; i < _nParticles * _nSplits; ++i)
	{
		int projectedXOffset = (int)(i / _nSplits) * _nSamples;
		// the current X is from projectedX[id] to projectedX[id+nSamples]
		int leftySize = 0;
		int rightySize = 0;

		for (int j = 0; j < _nClasses; ++j)
		{
			leftyProbs[j] = 0;
			rightyProbs[j] = 0;
		}

		for (int j = 0; j < _nSamples; ++j)
		{
			int maxIdx = 0;
			if (projectedX[projectedXOffset + j] <= splits[i])
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

		// find probabilties of each class
		for (int j = 0; j < _nClasses; ++j)
		{
			leftyProbs[j] /= (double)leftySize;
			rightyProbs[j] /= (double)rightySize;
		}

		// entropy of right and left
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
		// weighted cross entropy
		splitLosses[i] = (leftySize / (double)_nSamples) * leftyE +
						 (rightySize / (double)_nSamples) * rightyE;
	}
}

std::vector<double> ParticleSwarmOptimizer::generateSplit(Eigen::MatrixXd X)
{
	double Xmin = X.minCoeff();
	double Xmax = X.maxCoeff();
	double linspace = (Xmax - Xmin) / (_nSplits + 2);
	std::vector<double> split;
	for (int i = 0; i < _nSplits; ++i, Xmin += linspace)
		split.push_back(Xmin + linspace);
	return split;
}

// TODO: remove intermediate data structures
void ParticleSwarmOptimizer::getProjectionsSplits(double *projectedXBuf, double *splitsBuf)
{
	std::vector<Eigen::VectorXd> projectedX;
	std::vector<std::vector<double>> splits;

	// generate the A vectors and projected X vectors and splits in CPU
	// projectedX.shape = nSamples, 1
	for (int i = 0; i < _nParticles; ++i)
	{
		projectedX.push_back(_X * _particles[i]);
		splits.push_back(generateSplit(projectedX.back()));
	}

	// fill result buffers
	// can optimize into single for loop later
	for (int i = 0; i < _nParticles; ++i)
	{
		for (int j = i * _nSamples; j < (i + 1) * _nSamples; ++j)
			projectedXBuf[j] = projectedX[i].data()[j - i * _nSamples];
		for (int j = i * _nSplits; j < (i + 1) * _nSplits; ++j)
			splitsBuf[j] = splits[i][j - i * _nSplits];
	}
}

std::pair<double, double> ParticleSwarmOptimizer::threadCalculateSplitWELoss(Eigen::VectorXd projectedX, int particleIdx, std::vector<double> splits)
{
	double leftyProbs[100];
	double rightyProbs[100];
	double minLoss = DBL_MAX;
	double bestSplit = 0;
	for (double split : splits)
	{
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
		double weightedCE = (leftySize / (double)_nSamples) * leftyE + (rightySize / (double)_nSamples) * rightyE;
		if (weightedCE < minLoss)
		{
			minLoss = weightedCE;
			bestSplit = split;
		}
	}
	return std::pair<double, double>(minLoss, bestSplit);
}

std::pair<double, double> ParticleSwarmOptimizer::threadCalculateSplitSELoss(Eigen::VectorXd projectedX, int particleIdx, std::vector<double> splits)
{
	double minLoss = DBL_MAX;
	double bestSplit = 0;
	for (double split : splits)
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
		double weightedSE = (leftySize / (double)_nSamples) * leftyH + (rightySize / (double)_nSamples) * rightyH;
		if (weightedSE < minLoss)
		{
			minLoss = weightedSE;
			bestSplit = split;
		}
	}
	return std::pair<double, double>(minLoss, bestSplit);
}

// generate the new projectedX with particle and get splits and then calculate the losses
// update personal best and global best
int ParticleSwarmOptimizer::calculateBestLosses(int particleIdx)
{
	Eigen::VectorXd projectedX = _X * _particles[particleIdx];
	std::vector<double> splits = generateSplit(projectedX);
	std::pair<double, double> res;
	if (_nClasses > 1)
		res = threadCalculateSplitWELoss(projectedX, particleIdx, splits);
	else
		res = threadCalculateSplitSELoss(projectedX, particleIdx, splits);

	double loss = res.first;
	double split = res.second;

	if (loss < bestLosses[particleIdx])
	{
		bestLosses[particleIdx] = loss;
		bestSplits[particleIdx] = split;
		std::vector<double> stdParticle(_particles[particleIdx].data(), _particles[particleIdx].data() + _particles[particleIdx].rows() * _particles[particleIdx].cols());
		bestParticles[particleIdx] = stdParticle;

		if (loss < _globalBestLoss)
		{
			mtx.lock();
			_globalBestLoss = loss;
			_globalBestParticle = _particles[particleIdx];
			mtx.unlock();
		}
	}
	return 1;
}

// update velocities and particles
int ParticleSwarmOptimizer::updateVelocitiesParticles(int particleIdx)
{
	Eigen::VectorXd eigenBestParticle_i = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(bestParticles[particleIdx].data(), bestParticles[particleIdx].size());
	_velocities[particleIdx] = _w * _velocities[particleIdx] + _c1 * _r1 * (eigenBestParticle_i - _particles[particleIdx]) +
							   _c2 * _r2 * (_globalBestParticle - _particles[particleIdx]);
	_particles[particleIdx] = _particles[particleIdx] + _velocities[particleIdx];
	// clip and round
	for (int j = 0; j < _nFeatures; ++j)
	{
		if (_particles[particleIdx](j) > _minAn)
		{
			_particles[particleIdx](j) = _minAn;
		}
		else if (_particles[particleIdx][j] < -_minAn)
		{
			_particles[particleIdx](j) = -_minAn;
		}
		_particles[particleIdx](j) = round(_particles[particleIdx](j));
	}
	return 1;
}

void ParticleSwarmOptimizer::updateMulti()
{
	const int nThreads = std::min(_nParticles, (int)std::thread::hardware_concurrency());
	boost::asio::thread_pool pool(nThreads);
	int particleIdx;
	std::vector<std::future<int>> futures;
	for (int i = 0; i < _maxIter; ++i)
	{
		particleIdx = 0;
		while (particleIdx < _nParticles)
		{
			for (int j = 0; j < nThreads && particleIdx < _nParticles; ++j, ++particleIdx)
				futures.push_back(boost::asio::post(pool, std::packaged_task<int()>(std::bind(&ParticleSwarmOptimizer::calculateBestLosses, this, particleIdx))));
			for (int j = 0; j < nThreads && particleIdx < _nParticles; ++j, ++particleIdx)
				futures[j].wait();
			futures.clear();
		}
		particleIdx = 0;
		while (particleIdx < _nParticles)
		{
			for (int j = 0; j < nThreads && particleIdx < _nParticles; ++j, ++particleIdx)
				futures.push_back(boost::asio::post(pool, std::packaged_task<int()>(std::bind(&ParticleSwarmOptimizer::updateVelocitiesParticles, this, particleIdx))));
			for (int j = 0; j < nThreads && particleIdx < _nParticles; ++j, ++particleIdx)
				futures[j].wait();
			futures.clear();
		}
	}
	pool.join();
}

void ParticleSwarmOptimizer::update()
{
	double *projectedXBuf = (double *)malloc(sizeof(double) * _nParticles * _nSamples);
	double *splitsBuf = (double *)malloc(sizeof(double) * _nParticles * _nSplits);
	double *splitLossesBuf = (double *)malloc(sizeof(double) * _nParticles * _nSplits);
	for (int k = 0; k < _maxIter; ++k)
	{
		getProjectionsSplits(projectedXBuf, splitsBuf);
		if (_nClasses > 1) // classification
		{
			cpuCalculateSplitWELosses(projectedXBuf, splitsBuf, splitLossesBuf);
			// gpuCalculateSplitWELosses(_nParticles, _nSplits, _nFeatures, _nSamples, _nClasses,
			// 					projectedXBuf, splitsBuf, _y, splitLossesBuf);
		}
		else // regression
		{
			cpuCalculateSplitSELosses(projectedXBuf, splitsBuf, splitLossesBuf);
			// gpuCalculateSplitSELosses(_nParticles, _nSplits, _nFeatures, _nSamples,
			// 					projectedXBuf, splitsBuf, _y, splitLossesBuf);
		}
		// displayBuffer(splitLossesBuf, _nParticles, _nSplits);
		for (int i = 0; i < _nParticles; ++i)
		{
			for (int j = i * _nSplits; j < (i + 1) * _nSplits; ++j)
			{
				if (splitLossesBuf[j] < bestLosses[i])
				{
					bestLosses[i] = splitLossesBuf[j];
					bestSplits[i] = splitsBuf[j];
					std::vector<double> stdParticle(_particles[i].data(), _particles[i].data() + _particles[i].rows() * _particles[i].cols());
					bestParticles[i] = stdParticle;

					if (bestLosses[i] < _globalBestLoss)
					{
						_globalBestLoss = bestLosses[i];
						_globalBestParticle = _particles[i];
					}
				}
			}
		}

		for (int i = 0; i < _nParticles; ++i)
		{
			// update velocities and particles
			Eigen::VectorXd eigenBestParticle_i = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(bestParticles[i].data(), bestParticles[i].size());
			_velocities[i] = _w * _velocities[i] + _c1 * _r1 * (eigenBestParticle_i - _particles[i]) +
							 _c2 * _r2 * (_globalBestParticle - _particles[i]);
			_particles[i] = _particles[i] + _velocities[i];
			// clip and round
			for (int j = 0; j < _nFeatures; ++j)
			{
				if (_particles[i](j) > _minAn)
				{
					_particles[i](j) = _minAn;
				}
				else if (_particles[i][j] < -_minAn)
				{
					_particles[i](j) = -_minAn;
				}
				_particles[i](j) = round(_particles[i](j));
			}
		}
	}
	free(projectedXBuf);
	free(splitsBuf);
	free(splitLossesBuf);
}

/*
int main()
{
	const int nParticles = 300;
	const int nSplits = 64;
	const int nClasses = 1;
	const int nFeatures = 4;	 // assume we've already chosen a feature to split on
	const int nSamples = 1000; // this is the dimension of training data
	const int minAn = nFeatures;
	const int maxIter = 10;
	const double w = 0.7;
	const double c1 = 2;
	const double c2 = 2;
	const double r1 = 0.5;
	const double r2 = 0.5;
	Eigen::MatrixXd X = Eigen::MatrixXd::Random(nSamples, nFeatures);
	Eigen::MatrixXd y = Eigen::MatrixXd::Random(nSamples, nClasses);
	double *XSelect = (double *)malloc(sizeof(double) * nSamples * nFeatures);
	double *yBuf = (double *)malloc(sizeof(double) * nSamples * nClasses);
	for (int i = 0; i < nSamples; ++i)
	{
		for (int j = 0; j < nClasses; ++j)
			yBuf[i*nClasses+j] = (y(i, j)+1)/2;
		for (int j = 0; j < nFeatures; ++j)
			XSelect[i*nFeatures+j] = X(i, j);
	}
	ParticleSwarmOptimizer p1 = ParticleSwarmOptimizer(nParticles, nSplits, nFeatures, nClasses, nSamples, w, c1, c2, r1, r2, maxIter, minAn, XSelect, yBuf);
	ParticleSwarmOptimizer p2 = ParticleSwarmOptimizer(nParticles, nSplits, nFeatures, nClasses, nSamples, w, c1, c2, r1, r2, maxIter, minAn, XSelect, yBuf);
	auto start = std::chrono::high_resolution_clock::now();
	p1.update();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken by single thread: "
		 << duration.count() << " microseconds" << std::endl;
	start = std::chrono::high_resolution_clock::now();
	p2.update1();
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken by multi thread: "
		 << duration.count() << " microseconds" << std::endl;
	// for (int i = 0; i < nParticles; ++i)
	// {
	// 	for (int j = 0; j < nFeatures; ++j)
	// 	{
	// 		std::cout << p1.bestParticles[i][j] << " ";
	// 	}
	// 	std::cout << "\t";
	// 	for (int j = 0; j < nFeatures; ++j)
	// 	{
	// 		std::cout << p2.bestParticles[i][j] << " ";
	// 	}
	// 	std::cout << "\n";
	// }
	// std::cout << "splits" << std::endl;
	// for (int i = 0; i < nParticles; ++i)
	// 	std::cout << p1.bestSplits[i] << " " << p2.bestSplits[i] << std::endl;
	// std::cout << "losses" << std::endl;
	// for (int i = 0; i < nParticles; ++i)
	// 	std::cout << p1.bestLosses[i] << " " << p2.bestLosses[i] << std::endl;
}
*/
