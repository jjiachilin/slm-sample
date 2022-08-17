#include <cuda.h>
#include "math.h"
#include "stdio.h"
#include "time.h"
#include <iostream>
#include "split_loss.h"


#define N_BLOCKS 256
#define THREADS_PER_BLOCK 256

/*
each thread is responsible for one projection and one split_value
Params:
	projectedX: (nProjections * nSamples)
	y: (nSamples)
	splits: (nProjections * nSplits)
	splitLosses: (nProjecitons * nSplits)
*/
__global__ void __calculateSplitSELosses(double *projectedX, double *y, double *splits, double *splitLosses,
										 int nProjections, int nSamples, int nSplits)
{
	unsigned int splitOffset = blockIdx.x * blockDim.x + threadIdx.x;

	while (splitOffset < nProjections * nSplits)
	{
		unsigned int projectedXOffset = (unsigned int)(splitOffset / nSplits) * nSamples;
		// the current X is from projectedX[id] to projectedX[id+nSamples]
		unsigned int leftySize = 0;
		unsigned int rightySize = 0;
		double leftySum = 0;
		double rightySum = 0;

		for (unsigned int i = 0; i < nSamples; ++i)
		{
			if (projectedX[projectedXOffset + i] <= splits[splitOffset])
			{
				++leftySize;
				leftySum += y[i];
			}
			else
			{
				++rightySize;
				rightySum += y[i];
			}
		}

		double leftyMean = leftySum / leftySize;
		double rightyMean = rightySum / rightySize;
		double leftyH = 0;
		double rightyH = 0;

		for (unsigned int i = 0; i < nSamples; ++i)
		{
			if (projectedX[projectedXOffset + i] <= splits[splitOffset])
			{
				leftyH += (y[i] - leftyMean) * (y[i] - leftyMean);
			}
			else
			{
				rightyH += (y[i] - rightyMean) * (y[i] - rightyMean);
			}
		}
		// weighted squared error
		splitLosses[splitOffset] = (leftySize / (double)nSamples) * leftyH +
								   (rightySize / (double)nSamples) * rightyH;
		// go to next item in splits
		splitOffset += blockDim.x * gridDim.x;
	}
}
/*
each thread is responsible for one projection and one split_value
Params:
	projectedX: (nProjections * nSamples)
	y: (nSamples * nClasses)
	splits: (nProjections * nSplits)
	splitLosses: (nProjecitons * nSplits)
*/
__global__ void __calculateSplitWELosses(double *projectedX, double *y, double *splits, double *splitLosses,
										 int nProjections, int nSamples, int nSplits, const int nClasses)
{
	unsigned int splitOffset = blockIdx.x * blockDim.x + threadIdx.x;

	while (splitOffset < nProjections * nSplits)
	{
		unsigned int projectedXOffset = (unsigned int)(splitOffset / nSplits) * nSamples;
		// the current X is from projectedX[id] to projectedX[id+nSamples]
		unsigned int leftySize = 0;
		unsigned int rightySize = 0;
		double leftyProbs[100];
		double rightyProbs[100];

		for (unsigned int i = 0; i < nClasses; ++i)
		{
			leftyProbs[i] = 0;
			rightyProbs[i] = 0;
		}

		// split y by X
		for (unsigned int i = 0; i < nSamples; ++i)
		{
			unsigned int maxIdx = 0;
			if (projectedX[projectedXOffset + i] <= splits[splitOffset])
			{
				++leftySize;
				for (unsigned int j = 1; j < nClasses; ++j)
				{
					if (y[i * nClasses + j] > y[i * nClasses + maxIdx])
					{
						maxIdx = j;
					}
				}
				++leftyProbs[maxIdx];
			}
			else
			{
				++rightySize;
				for (unsigned int j = 1; j < nClasses; ++j)
				{
					if (y[i * nClasses + j] > y[i * nClasses + maxIdx])
					{
						maxIdx = j;
					}
				}
				++rightyProbs[maxIdx];
			}
		}

		// find probabilties of each class
		for (unsigned int i = 0; i < nClasses; ++i)
		{
			leftyProbs[i] /= (double)leftySize;
			rightyProbs[i] /= (double)rightySize;
		}

		// entropy of right and left
		double rightyE = 0;
		double leftyE = 0;

		for (unsigned int i = 0; i < nClasses; ++i)
		{
			if (rightyProbs[i] > 0)
				rightyE -= rightyProbs[i] * log(rightyProbs[i]);
			if (leftyProbs[i] > 0)
				leftyE -= leftyProbs[i] * log(leftyProbs[i]);
		}
		rightyE /= log10((double)nClasses);
		leftyE /= log10((double)nClasses);

		// weighted cross entropy
		splitLosses[splitOffset] = ((double)leftySize / (double)nSamples) * leftyE +
								   ((double)rightySize / (double)nSamples) * rightyE;
		// go to next item in splits
		splitOffset += blockDim.x * gridDim.x;
	}
}

void gpuCalculateSplitSELosses(int nProjections, int nSplits,
							   int nFeatures, int nSamples,
							   double *projectedXBuf, double *splitsBuf,
							   const double *yBuf, double *splitLossesBuf)
{
	// allocate device memory
	double *d_projectedXBuf, *d_splitsBuf, *d_splitLossesBuf, *d_yBuf;
	cudaMalloc((void **)&d_yBuf, sizeof(double) * nSamples);
	cudaMalloc((void **)&d_projectedXBuf, sizeof(double) * nProjections * nSamples);
	cudaMalloc((void **)&d_splitsBuf, sizeof(double) * nProjections * nSplits);
	cudaMalloc((void **)&d_splitLossesBuf, sizeof(double) * nProjections * nSplits);

	// copy data to device
	cudaMemcpy(d_yBuf, yBuf, sizeof(double) * nSamples, cudaMemcpyHostToDevice);
	cudaMemcpy(d_projectedXBuf, projectedXBuf, sizeof(double) * nProjections * nSamples, cudaMemcpyHostToDevice);
	cudaMemcpy(d_splitsBuf, splitsBuf, sizeof(double) * nProjections * nSplits, cudaMemcpyHostToDevice);

	// call kernel
	__calculateSplitSELosses<<<N_BLOCKS, THREADS_PER_BLOCK>>>(d_projectedXBuf, d_yBuf,
															  d_splitsBuf, d_splitLossesBuf,
															  nProjections, nSamples, nSplits);

	// copy results to host
	cudaMemcpy(splitLossesBuf, d_splitLossesBuf, sizeof(double) * nProjections * nSplits, cudaMemcpyDeviceToHost);
	// free device memory
	cudaFree(d_yBuf);
	cudaFree(d_projectedXBuf);
	cudaFree(d_splitsBuf);
	cudaFree(d_splitLossesBuf);
}

void gpuCalculateSplitWELosses(int nProjections, int nSplits,
							   int nFeatures, int nSamples,
							   int nClasses, double *projectedXBuf,
							   double *splitsBuf, const double *yBuf, double *splitLossesBuf)
{
	// allocate device memory
	double *d_projectedXBuf, *d_splitsBuf, *d_splitLossesBuf, *d_yBuf;
	cudaMalloc((void **)&d_yBuf, sizeof(double) * nSamples * nClasses);
	cudaMalloc((void **)&d_projectedXBuf, sizeof(double) * nProjections * nSamples);
	cudaMalloc((void **)&d_splitsBuf, sizeof(double) * nProjections * nSplits);
	cudaMalloc((void **)&d_splitLossesBuf, sizeof(double) * nProjections * nSplits);

	// copy data to device
	cudaMemcpy(d_yBuf, yBuf, sizeof(double) * nSamples * nClasses, cudaMemcpyHostToDevice);
	cudaMemcpy(d_projectedXBuf, projectedXBuf, sizeof(double) * nProjections * nSamples, cudaMemcpyHostToDevice);
	cudaMemcpy(d_splitsBuf, splitsBuf, sizeof(double) * nProjections * nSplits, cudaMemcpyHostToDevice);

	// call kernel
	__calculateSplitWELosses<<<N_BLOCKS, THREADS_PER_BLOCK>>>(d_projectedXBuf, d_yBuf,
															  d_splitsBuf, d_splitLossesBuf,
															  nProjections, nSamples, nSplits, nClasses);

	// copy results to host
	cudaMemcpy(splitLossesBuf, d_splitLossesBuf, sizeof(double) * nProjections * nSplits, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_yBuf);
	cudaFree(d_projectedXBuf);
	cudaFree(d_splitsBuf);
	cudaFree(d_splitLossesBuf);
}
