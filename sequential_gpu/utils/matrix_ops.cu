#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "matrix_ops.h"

int* load_matrix(const char* filename, int* width, int* height)
{
	int* mat;
	int i,j,w,h;
	FILE* f;

	f = fopen(filename, "r");

	if(width != NULL && height != NULL)
	{
		if(fscanf(f, "%d %d", &h, &w) != 2)
		{
			printf("Error de lectura de dimensiones de matriz.\n");
			abort();
		}
	}
	else //mask load case
		h = w = MASK_DIM;

	mat = (int*)malloc(h*w*sizeof(int));

	for(i=0; i<h; i++)
		for(j=0; j<w; j++)
		{	
			if(fscanf(f, "%d", &mat[i*w + j]) != 1)
			{
				printf("Error de lectura de datos de matriz.\n");
				abort();
			}
			fgetc(f);
		}

	if(width != NULL && height != NULL)
	{
		*width = w;
		*height = h;
	}
	return mat;
}

int* load_mask(const char* filename)
{
	return load_matrix(filename, NULL, NULL);
}

void save_matrix(const char* filename, int* matrix, int width, int height)
{
	int i,j;
	FILE* f;

	f = fopen(filename, "w");
	
	fprintf(f, "%d %d\n", height, width);
	for(i=0; i<height; i++)
	{
		for(j=0; j<width; j++)
		{
			fprintf(f, "%d", matrix[i*width + j]);
			if(j<width-1)
				fprintf(f, " ");
		}
		fprintf(f, "\n");
	}

	fclose(f);
}

int* mmalloc(int height, int width)
{
	int* matrix = NULL;

	if((matrix=(int*)malloc(height*width*sizeof(int))) != NULL)
		return matrix;
	else
	{
		perror("Memory allocation error.\n");
		exit(-1);
	}
}
void mfree(int* matrix)
{
	free(matrix);
}

__global__ void gpu_noise_init(uint seed, curandState_t* states, int n)
{
	for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<n; i += blockDim.x * gridDim.x)
		curand_init((seed<<20)+i, 0, 0, &states[i]);
}

__global__ void gpu_noise_maker(curandState_t* states, int* matrix, int* noisy_matrix, float mu, float sigma, int n)
{
	for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<n; i += blockDim.x * gridDim.x)
	{
		if(sigma != 0)
		{
			noisy_matrix[i] = matrix[i] * (mu + curand_normal(&states[i])*sigma);
			if(noisy_matrix[i] > 255)
				noisy_matrix[i] = 255;
		}
		else
			noisy_matrix[i] = matrix[i];
	}
}