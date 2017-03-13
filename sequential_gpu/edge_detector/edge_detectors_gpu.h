#ifndef _Edge_Detectors_GPU_H
#define _Edge_Detectors_GPU_H

#include <cuda.h>

__global__ void gpu_edge_detector_g(int* img, int* edge, int width, int height, int* mask);
__global__ void gpu_edge_detector_cv(int* img, int* edge, int width, int height);

#endif