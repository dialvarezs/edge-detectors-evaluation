#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "matrix_ops.h"

int* load_matrix(const char* filename, int* width, int* height)
{
	int* mat;
	int w,h;
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

	mat = malloc(h*w*sizeof(int));

	for(int i=0; i<h; i++)
		for(int j=0; j<w; j++)
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
	FILE* f;

	f = fopen(filename, "w");
	
	fprintf(f, "%d %d\n", height, width);
	for(int i=0; i<height; i++)
	{
		for(int j=0; j<width; j++)
		{
			fprintf(f, "%d", matrix[i*width + j]);
			if(j<width-1)
				fprintf(f, " ");
		}
		fprintf(f, "\n");
	}

	fclose(f);
}

void fill_borders(int* matrix, int width, int height)
{
	#pragma omp parallel for shared(matrix, width, height)
		for(int i=1; i<height-1; i++)
		{
			matrix[i*width] = matrix[i*width + 1];
			matrix[(i+1)*width-1] = matrix[(i+1)*width-2];
		}
	#pragma omp parallel for shared(matrix, width, height)
		for(int i=0; i<width; i++)
		{
			matrix[i] = matrix[width + i];
			matrix[width*(height-1) + i] = matrix[width*(height-1) + i];
		}
}

void noise_maker_multiplicative(int* matrix, int* noisy_matrix, int h, int w, float s)
{
	int max_threads = omp_get_max_threads();
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	grstg* storages = malloc(max_threads*sizeof(grstg));
	for(int i=0; i<max_threads; i++)
		storages[i].state = 0;

	#pragma omp parallel for shared(storages)
		for(int i=0; i<h*w; i++)
		{
			int thread = omp_get_thread_num();
			if(s != 0)
			{
				noisy_matrix[i] = matrix[i]*gaussrand(s, 1, &storages[thread], tm.tm_sec*thread+tm.tm_mday*tm.tm_yday);
				if(noisy_matrix[i] > 255)
					noisy_matrix[i] = 255;
			}
			else
			noisy_matrix[i] = matrix[i];
		}
}

/*
	Based on http://c-faq.com/lib/gaussian.html
	Uses a structure to save the extra value and avoid the use of static vars 
*/
float gaussrand(float sigma, float mu, grstg* stg, unsigned seed)
{
	float X, V1, V2, U1, U2, S;

	if(stg->state == 0)
	{
		do {
			U1 = (float)rand_r(&seed) / RAND_MAX;
			U2 = (float)rand_r(&seed) / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
		stg->V = V2 * sqrt(-2 * log(S) / S);
	}
	else
		X = stg->V;

	stg->state = 1 - stg->state;

	return mu + X * sigma;
}

int* mmalloc(int height, int width)
{
	int* matrix = NULL;

	if((matrix=malloc(height*width*sizeof(int))) != NULL)
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