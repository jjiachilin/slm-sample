#ifndef SPLIT_LOSS_H
#define SPLIT_LOSS_H

void gpuCalculateSplitSELosses(int nProjections, int nSplits,
							   int nFeatures, int nSamples,
							   double *projectedXBuf, double *splitsBuf,
							   const double *yBuf, double *splitLossesBuf);

void gpuCalculateSplitWELosses(int nProjections, int nSplits,
							   int nFeatures, int nSamples,
							   int nClasses, double *projectedXBuf,
							   double *splitsBuf, const double *yBuf, double *splitLossesBuf);

#endif
