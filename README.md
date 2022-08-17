# SLM Sample

This repo is to showcase a part of my work I did at USC Media Communications Lab
under Professor C-C. Jay Kuo for the summer of 2022. I implemented an efficient 
version of a novel decision tree based algorithm called the Subspace Learning
Machine (SLM) [1]. 

In summary, SLM projects the sample space at each node into a more discriminant
subspace so that we can obtain shallower trees with the same performance. The
main bottleneck of the code is trying to find the projection vector. 

First, I changed the original Python code to Cython and then integrated a C++ 
extension for the bottleneck portions of the algorithm. I implemented and 
tested 2 methods for finding the best projection vector: a probabilistic 
selection method detailed in the above paper, and Adaptive Particle Swarm 
Optimization (APSO) [2]. For probabilistic selection, I designed and tested
single-threaded, multi-threaded, and GPU implementations. For APSO, I did the same for 
single-threaded and multi-threaded versions. 

Ultimately, the GPU version of probabilistic selection achieved a x700 speedup from 
the original pure Python code while the single threaded APSO version had similar
runtime performance. Multithreaded APSO did not provide a significant speedup
for a reasonable range for the number of particles used in the optimization process.

[1] Fu, Hongyu; Yang, Yijing; Mishra, Vinod K.; Kuo, Jay C.-C. "Subspace Learning Machine (SLM): Methodology and Performance" https://arxiv.org/pdf/2205.05296.pdf
[2] Zhan, Z-H.; Zhang, J.; Li, Y.; Chung, H.S-H. "Adaptive particle swarm optimization" https://eprints.gla.ac.uk/7645/1/7645.pdf 
