#ifndef _Matrix_Ops_GPU_H
#define _Matrix_Ops_GPU_H

#include <curand.h>
#include <curand_kernel.h>

#define MASK_DIM 3

// made for saving the extra value of the Gauss rand generator
struct _gaussrandstorage
{
	int state;
	float V;
};
typedef struct _gaussrandstorage grstg;


int* load_matrix(const char* filename, int* width, int* height);
int* load_mask(const char* filename);
void save_matrix(const char* filename, int* matrix, int width, int height);
void fill_borders(int* matrix, int width, int height);
void noise_maker_saltpepper(int* matrix, int* noisy_matrix, int h, int w, float q);
void noise_maker_multiplicative(int* matrix, int* noisy_matrix, int h, int w, float s, unsigned seed);
void noise_maker_additive(int* matrix, int* noisy_matrix, int h, int w, float s, unsigned seed);
float gaussrand(float sigma, float mu, grstg* stg, unsigned seed);
int* mmalloc(int height, int width);
void mfree(int* matrix);
int median_qselect(int* values, int n);
void swap(int* a, int* b);

__global__ void gpu_noise_init(uint seed, curandState_t* states, int n);
__global__ void gpu_noise_maker(curandState_t* states, int* matrix, int* noisy_matrix, float mu, float sigma, int n);

#endif