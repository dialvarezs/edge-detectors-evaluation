#include <stdio.h>
#include "../utils/matrix_ops.h"
#include "../utils/gpu_consts.h"

void usage();


int main(int argc, char** argv)
{
	int* matrix = NULL;
	int* noisy_matrix = NULL;
	int* dev_matrix = NULL;
	int* dev_noisy_matrix = NULL;
	int h, w, size;
	float sigma;
	curandState_t* states = NULL;
	
	if(argc < 4)
	{
		usage();
		return -1;
	}

	matrix = load_matrix(argv[1], &w, &h);
	sigma = atof(argv[3]);

	size = w*h*sizeof(int);
	noisy_matrix =  mmalloc(h, w);
	cudaMalloc(&dev_matrix, size);
	cudaMalloc(&dev_noisy_matrix, size);
	cudaMalloc(&states, h*w*sizeof(curandState_t));

	cudaMemcpy(dev_matrix, matrix, size, cudaMemcpyHostToDevice);

	gpu_noise_init<<<BLOCKS, THREADS>>>(time(0), states, h*w);
	gpu_noise_maker<<<BLOCKS, THREADS>>>(states, dev_matrix, dev_noisy_matrix, 1.0, sigma, h*w);


	cudaMemcpy(noisy_matrix, dev_noisy_matrix, size, cudaMemcpyDeviceToHost);

	save_matrix(argv[2], noisy_matrix, w, h);

	mfree(matrix);
	mfree(noisy_matrix);
	cudaFree(dev_matrix);
	cudaFree(dev_noisy_matrix);
	cudaFree(states);


	return 0;
}


void usage()
{
	printf(
		"Usage:\n"
		"\t./noise_maker <matrix_file> <noisy_matrix_file(out)> <sigma>\n"
		);
}