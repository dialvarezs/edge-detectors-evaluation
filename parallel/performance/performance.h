#ifndef _Performance_H
#define _Performance_H

typedef float (*comparison_index)(int*, int*, int, int, int);

void find_threshold_exhaustive(int* edge, int* ground_truth, int width, int height, comparison_index comparison, int* threshold, float* similarity);
void find_threshold_optimized(int iteration, int initval, int delta, int tolerance, float K,
	int* edge, int* ground_truth, int width, int height, comparison_index comparison, int* threshold, float* similarity);
float edge_comparison(int* edge, int* ground_truth, int threshold, int width, int height);
int* binarization(int* edge, int* binarized, int width, int height, int threshold);
int* histogram(int* matrix, int width, int height);

#endif