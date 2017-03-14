#ifndef _Matrix_Ops_H
#define _Matrix_Ops_H

#include <curand.h>
#include <curand_kernel.h>

#define MASK_DIM 3

int* load_matrix(const char* filename, int* width, int* height);
int* load_mask(const char* filename);
void save_matrix(const char* filename, int* matrix, int width, int height);
int* mmalloc(int height, int width);
void mfree(int* matrix);

__global__ void gpu_noise_init(uint seed, curandState_t* states, int n);
__global__ void gpu_noise_maker(curandState_t* states, int* matrix, int* noisy_matrix, float mu, float sigma, int n);

#endif