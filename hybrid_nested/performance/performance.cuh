#ifndef _Performance_GPU_H
#define _Performance_GPU_H

#include <cuda.h>

typedef float (*comparison_index)(int*, int*, int, int, int);

void find_threshold_exhaustive(int* edge, int* ground_truth, int width, int height, comparison_index comparison, int* threshold, float* similarity);
void find_threshold_optimized(int iteration, int initval, int delta, int tolerance, float K,
	int* edge, int* ground_truth, int width, int height, comparison_index comparison, int* threshold, float* similarity);
float edge_comparison(int* edge, int* ground_truth, int threshold, int width, int height);
int* binarization(int* edge, int* binarized, int width, int height, int threshold);
int* histogram(int* matrix, int* hist, int width, int height);

void gpu_find_threshold_exhaustive(int* edge, int* ground_truth, int width, int height, comparison_index comparison, int* threshold, float* similarity);
void gpu_find_threshold_optimized(int iteration, int initval, int delta, int tolerance, float K,
	int* edge, int* ground_truth, int width, int height, comparison_index comparison, int* threshold, float* similarity);
float gpu_edge_comparison(int* edge, int* ground_truth, int threshold, int width, int height);
int* gpu_histogram(int* matrix, int width, int height);

// GPU code
__global__ void gpu_binarization(int* gpu_edge, int* gpu_binarized, int width, int height, int threshold);
__inline__ __device__ int3 warp_reduce_sum_triple(int3 vals);
__global__ void device_edge_comparison(int *edge, int* gt, int* success, int* false_pos, int* false_neg, int n);
__global__ void device_histogram(int* matrix, int* hist, int n);


#endif