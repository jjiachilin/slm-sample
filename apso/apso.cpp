#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unordered_map>
#include <random>
#include <numeric>
#include <iostream>
#include "math.h"
#include "apso.h"
#include "float.h"

APSO::APSO(int nParticles, int nSplits, int nFeatures, int nClasses, int nSamples,
		   double w, int maxIter, int minAn, double c1, double c2, double percent,
		   double stdMin, double stdMax, double *X, double *y)
	: _nParticles(nParticles), _nSplits(nSplits), _nFeatures(nFeatures), _nClasses(nClasses), _nSamples(nSamples), _w(w), _maxIter(maxIter),
	  _minAn(minAn), _vMax(percent * minAn), _c1(c1), _c2(c2), _stdMin(stdMin), _stdMax(stdMax), _y(y)
{
	_state = 1;
	_X = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(X, nSamples, nFeatures);
	std::uniform_real_distribution<double> uniform01(0, 1);
	for (int i = 0; i < nParticles; ++i)
	{
		Eigen::VectorXd Avec(nFeatures);
		Eigen::VectorXd vel(nFeatures);
		for (int j = 0; j < nFeatures; ++j)
		{
			Avec(j) = -minAn + rand() % (int)((minAn + 1) - (-minAn));
			vel(j) = uniform01(_generator) * _vMax;
		}
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
		_velocities.push_back(vel);
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
		cpuCalculateSplitWELosses(projectedXBuf, splitsBuf, splitLossesBuf, _nParticles);
	}
	else // regression
	{
		// gpuCalculateSplitSELosses(_nParticles, _nSplits, _nFeatures, _nSamples,
		// 					projectedXBuf, splitsBuf, _y, splitLossesBuf);
		cpuCalculateSplitSELosses(projectedXBuf, splitsBuf, splitLossesBuf, _nParticles);
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
					_globalBestParticleIdx = i;
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

void APSO::cpuCalculateSplitSELosses(double *projectedX, double *splits, double *splitLosses, int nParticles)
{
	for (int i = 0; i < nParticles * _nSplits; ++i)
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

void APSO::cpuCalculateSplitWELosses(double *projectedX, double *splits, double *splitLosses, int nParticles)
{
	double leftyProbs[100]; // hopefully we won't have to predict more than 100 classes in the near future
	double rightyProbs[100];

	for (int i = 0; i < nParticles * _nSplits; ++i)
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

std::pair<double, double> APSO::threadCalculateSplitWELoss(Eigen::VectorXd projectedX, std::vector<double> splits)
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

std::pair<double, double> APSO::threadCalculateSplitSELoss(Eigen::VectorXd projectedX, std::vector<double> splits)
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

std::vector<double> APSO::generateSplit(Eigen::MatrixXd X)
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
void APSO::getProjectionsSplits(double *projectedXBuf, double *splitsBuf)
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

std::vector<double> APSO::euclidDist(Eigen::VectorXd u, std::vector<Eigen::VectorXd> v)
{
	std::vector<Eigen::VectorXd> d;
	std::vector<double> r;
	for (auto vec : v)
		d.push_back(u - vec);
	for (int i = 0; i < d.size(); ++i)
		for (int j = 0; j < u.size(); ++j)
			d[i][j] = d[i][j] * d[i][j];
	for (int i = 0; i < d.size(); ++i)
		r.push_back(d[i].sum());
	return r;
}

std::vector<double> APSO::calDistance()
{
	std::vector<double> distList;
	for (int i = 0; i < _nParticles; ++i)
	{
		std::vector<Eigen::VectorXd> v;
		for (int j = 0; j < _nParticles; ++j)
			if (i != j)
				v.push_back(_particles[j]);
		std::vector<double> dist = euclidDist(_particles[i], v);
		distList.push_back(std::accumulate(dist.begin(), dist.end(), 0.0) / dist.size());
	}
	auto minmax = std::minmax_element(distList.begin(), distList.end());
	std::vector<double> ret{*(minmax.first),
							*(minmax.second),
							distList[_globalBestParticleIdx]};

	return ret;
}

double APSO::calF(double dMax, double dMin, double dBest)
{
	return (dBest - dMin) / (dMax - dMin);
}

void APSO::identifyState(double f)
{
	std::vector<double> s{f1(f), f2(f), f3(f), f4(f)};
	std::unordered_map<double, int> sCounter;
	for (auto d : s)
		s[d] += 1;

	if (sCounter[1] == 1)
	{
		_state = std::find(s.begin(), s.end(), 1) - s.begin() + 1;
	}
	else
	{
		int prevState = _state;
		if (sCounter[0] == 3)
		{
			for (int i = 0; i < 4; ++i)
				if (s[i] != 0)
					_state = i + 1;
		}
		else if (sCounter[0] == 2)
		{
			int s1 = -1;
			int s2 = -1;
			for (int i = 0; i < 4; ++i)
			{
				if (s[i] != 0 && s1 == -1)
					s1 = i + 1;
				else if (s[i] != 0 && s1 != -1)
					s2 = i + 1;
			}
			if (s1 == 1 && s2 == 2)
			{
				if (prevState == 1)
					_state = 1;
				else if (prevState == 4)
					_state = 1;
				else
					_state = 2;
			}
			else if (s1 == 1 && s2 == 4)
			{
				if (prevState == 1)
					_state = 1;
				else if (prevState == 2)
					_state = 1;
				else
					_state = 4;
			}
			else if (s1 == 2 && s2 == 3)
			{
				if (prevState == 2)
					_state = 2;
				else if (prevState == 1)
					_state = 2;
				else
					_state = 3;
			}
		}
	}
}

double APSO::f1(double f)
{
	double r = -1;
	if ((f >= 0) && (f <= 0.4))
		r = 0;
	else if ((f > 0.4) && (f <= 0.6))
		r = 5 * f - 2;
	else if ((f > 0.6) && (f <= 0.7))
		r = 1;
	else if ((f > 0.7) && (f <= 0.8))
		r = -10 * f + 8;
	else if ((f > 0.8) && (f <= 1))
		r = 0;
	return r;
}

double APSO::f2(double f)
{
	double r = -1;
	if ((f >= 0) && (f <= 0.2))
		r = 0;
	else if ((f > 0.2) && (f <= 0.3))
		r = 10 * f - 2;
	else if ((f > 0.3) && (f <= 0.4))
		r = 1;
	else if ((f > 0.4) && (f <= 0.6))
		r = -5 * f + 3;
	else if ((f > 0.6) && (f <= 1))
		r = 0;
	return r;
}

double APSO::f3(double f)
{
	double r = -1;
	if ((f >= 0) && (f <= 0.1))
		r = 1;
	else if ((f > 0.1) && (f <= 0.3))
		r = 5 * f - 1.5;
	else if ((f > 0.3) && (f <= 1))
		r = 0;
	return r;
}

double APSO::f4(double f)
{
	double r = -1;
	if ((f >= 0) && (f <= 0.7))
		r = 1;
	else if ((f > 0.7) && (f <= 0.9))
		r = 5 * f - 3.5;
	else if ((f > 0.9) && (f <= 1))
		r = 1;
	return r;
}

void APSO::updateC(int iter)
{
	std::uniform_real_distribution<double> uniformDist(0.05, 0.1);
	double n1 = uniformDist(_generator);
	double n2 = uniformDist(_generator);
	std::uniform_real_distribution<double> n1UniformDist(0, n1);
	std::uniform_real_distribution<double> n2UniformDist(0, n2);
	double d1 = n1UniformDist(_generator);
	double d2 = n2UniformDist(_generator);

	if (_state == 1)
	{
		_c1 += d1;
		_c2 -= d2;
	}
	else if (_state == 2)
	{
		_c1 += d1 * 0.5;
		_c2 -= d2 * 0.5;
	}
	else if (_state == 3)
	{
		_c1 += d1 * 0.5;
		_c2 += d2 * 0.5;
		doEliteLearning(iter);
	}
	else if (_state == 4)
	{
		_c1 -= d1;
		_c2 += d2;
	}

	if (_c2 > 2.5)
		_c2 = 2.5;
	else if (_c2 < 1.5)
		_c2 = 1.5;

	double tmp1 = _c1;
	double tmp2 = _c2;
	if (_c1 + _c2 > 4.0)
	{
		_c1 = tmp1 / (tmp1 + tmp2) * 4.0;
		_c2 = tmp2 / (tmp1 + tmp2) * 4.0;
	}
}

Eigen::VectorXd APSO::clipVec(Eigen::VectorXd particle, double least, double greatest)
{
	for (int j = 0; j < _nFeatures; ++j)
	{
		if (particle(j) > greatest)
		{
			particle(j) = greatest;
		}
		else if (particle(j) < least)
		{
			particle(j) = least;
		}
	}
	return particle;
}

Eigen::VectorXd APSO::roundVec(Eigen::VectorXd particle)
{
	for (int i = 0; i < _nFeatures; ++i)
		particle(i) = round(particle(i));
	return particle;
}

void APSO::doEliteLearning(int iter)
{
	Eigen::VectorXd P = _globalBestParticle;
	std::uniform_int_distribution<int> intUniformDist(0, _nFeatures);
	int d = intUniformDist(_generator);
	double std = _stdMax - (_stdMax - _stdMin) * (iter / _maxIter);
	std::normal_distribution<double> gaussian(0, std);
	double noise = gaussian(_generator);
	P(d) = P(d) + 2 * _minAn * noise;
	P = clipVec(P, -_minAn, _minAn);
	P = roundVec(P);
	// call loss function on P to find best loss and best split
	double *projectedXBuf = (double *)malloc(sizeof(double) * 1 * _nSamples);
	double *splitsBuf = (double *)malloc(sizeof(double) * 1 * _nSplits);
	double *splitLossesBuf = (double *)malloc(sizeof(double) * 1 * _nSplits);

	Eigen::VectorXd projectedX = _X * P;
	std::vector<double> splits = generateSplit(projectedX);

	std::pair<double, double> res;
	if (_nClasses > 1)
		res = threadCalculateSplitWELoss(projectedX, splits);
	else
		res = threadCalculateSplitSELoss(projectedX, splits);

	double loss = res.first;
	double split = res.second;

	if (loss < _globalBestLoss)
	{
		_particles[_globalBestParticleIdx] = P;
		std::vector<double> stdP(P.data(), P.data() + P.rows() * P.cols());
		bestParticles[_globalBestParticleIdx] = stdP;
		bestLosses[_globalBestParticleIdx] = loss;
		bestSplits[_globalBestParticleIdx] = split;
		_globalBestParticle = P;
		_globalBestLoss = loss;
	}
	else
	{
		int worstIdx = 0;
		for (int i = 0; i < _nParticles; ++i)
			if (bestLosses[i] > bestLosses[worstIdx])
				worstIdx = i;
		if (bestLosses[worstIdx] > loss)
		{
			_particles[worstIdx] = P;
			std::vector<double> stdP(P.data(), P.data() + P.rows() * P.cols());
			bestParticles[worstIdx] = stdP;
			bestLosses[worstIdx] = loss;
			bestSplits[worstIdx] = split;
		}
	}
	free(projectedXBuf);
	free(splitsBuf);
	free(splitLossesBuf);
}

void APSO::updateInertia(double f)
{
	_w = 1 / (1 + 1.5 * exp(-2.6 * f));
}

void APSO::updateParticles()
{
	std::uniform_real_distribution<double> uniformDist(0, 1);
	for (int i = 0; i < _nParticles; ++i)
	{
		const double r1 = uniformDist(_generator);
		const double r2 = uniformDist(_generator);
		// update velocity
		Eigen::VectorXd eigenBestParticle_i = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(bestParticles[i].data(), bestParticles[i].size());
		_velocities[i] = _w * _velocities[i] + _c1 * r1 * (eigenBestParticle_i - _particles[i]) +
						 _c2 * r2 * (_globalBestParticle - _particles[i]);
		_velocities[i] = clipVec(_velocities[i], -_vMax, _vMax);
		_particles[i] = _particles[i] + _velocities[i];
		_particles[i] = clipVec(_particles[i], -_minAn, _minAn);
		_particles[i] = roundVec(_particles[i]);
	}

	double *projectedXBuf = (double *)malloc(sizeof(double) * _nParticles * _nSamples);
	double *splitsBuf = (double *)malloc(sizeof(double) * _nParticles * _nSplits);
	double *splitLossesBuf = (double *)malloc(sizeof(double) * _nParticles * _nSplits);

	getProjectionsSplits(projectedXBuf, splitsBuf);
	if (_nClasses > 1) // classification
	{
		cpuCalculateSplitWELosses(projectedXBuf, splitsBuf, splitLossesBuf, _nParticles);
		// gpuCalculateSplitWELosses(_nParticles, _nSplits, _nFeatures, _nSamples, _nClasses,
		// 					projectedXBuf, splitsBuf, _y, splitLossesBuf);
	}
	else // regression
	{
		cpuCalculateSplitSELosses(projectedXBuf, splitsBuf, splitLossesBuf, _nParticles);
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
					_globalBestParticleIdx = i;
					_globalBestParticle = _particles[i];
				}
			}
		}
	}
	free(projectedXBuf);
	free(splitsBuf);
	free(splitLossesBuf);
}

void APSO::run()
{
	for (int i = 0; i < _maxIter; ++i)
	{
		std::vector<double> d = calDistance();
		double f = calF(d[0], d[1], d[2]);
		identifyState(f);
		updateC(i + 1);
		updateInertia(f);
		updateParticles();
	}
	std::cout << "minloss: " << _globalBestLoss << std::endl;
}
/*
int main()
{
	const int nParticles = 3;
	const int nSplits = 32;
	const int nClasses = 1;
	const int nFeatures = 4;	 // assume we've already chosen a feature to split on
	const int nSamples = 1000; // this is the dimension of training data
	const int minAn = nFeatures;
	const int maxIter = 10;
	const double w = 0.7;
	const double c1 = 2;
	const double c2 = 2;
	Eigen::MatrixXd X = Eigen::MatrixXd::Random(nSamples, nFeatures);
	Eigen::MatrixXd y = Eigen::MatrixXd::Random(nSamples, nClasses);
	double *XSelect = (double *)malloc(sizeof(double) * nSamples * nFeatures);
	double *yBuf = (double *)malloc(sizeof(double) * nSamples * nClasses);
	for (int i = 0; i < nSamples; ++i)
	{
		for (int j = 0; j < nClasses; ++j)
		{
			if (i > nSamples/2)
				yBuf[i*nClasses+j] = 1;
			else
				yBuf[i*nClasses+j] = 0;
		}
		for (int j = 0; j < nFeatures; ++j)
			XSelect[i*nFeatures+j] = j;
	}
	APSO p1 = APSO(nParticles, nSplits, nFeatures, nClasses, nSamples, w, maxIter, minAn, c1, c2, 0.2, 0.1, 1.0, XSelect, yBuf);
	p1.run();
	for (auto l : p1.lossCurve)
		std::cout << l << " " << std::endl;
	free(XSelect);
	free(yBuf);
}*/
