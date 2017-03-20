#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <libgen.h>
#include <stdbool.h>
#include <omp.h>
// #include <sys/stat.h>
#include <cuda.h>
#include "utils/matrix_ops.cuh"
#include "utils/gpu_consts.cuh"
#include "utils/vars.h"
#include "edge_detector/edge_detectors.cuh"
#include "performance/performance.cuh"


#define BILLION 1E9
#define timeit(before, after, f, ...) {\
	clock_gettime(CLOCK_MONOTONIC, &before);\
	f(__VA_ARGS__);\
	clock_gettime(CLOCK_MONOTONIC, &after);\
}
#define timeit_gpu(before, after, f, ...) {\
	cudaDeviceSynchronize();\
	clock_gettime(CLOCK_MONOTONIC, &before);\
	f(__VA_ARGS__);\
	cudaDeviceSynchronize();\
	clock_gettime(CLOCK_MONOTONIC, &after);\
}
#define timeit_gpu_kernel(before, after, k, ...) {\
	cudaDeviceSynchronize();\
	clock_gettime(CLOCK_MONOTONIC, &before);\
	k<<<BLOCKS, THREADS>>>(__VA_ARGS__);\
	cudaDeviceSynchronize();\
	clock_gettime(CLOCK_MONOTONIC, &after);\
}

#define SMIN 0.002
#define SMAX 0.202
#define STEPS 5
#define REPS 3

struct execution
{
	float sigma;
	int rep;
	bool use_gpu;
	float time_ms;
	float time_noise;
	float time_edge;
	float time_perf;
};
typedef struct execution execnode;


void usage();
char* name(char* path);
float time_diff(struct timespec before, struct timespec after);

int main(int argc, char** argv)
{
	int* matrix = NULL;
	int* ground_truth = NULL;
	int* noisy_matrix = NULL;
	int* edge = NULL;
	int** dev_matrix = NULL;
	int** dev_ground_truth = NULL;
	int** dev_noisy_matrices = NULL;
	int** dev_edges = NULL;
	curandState_t** states = NULL;
	execnode* exec_list = NULL;

	int w, h, size, threshold_cv, ncpu, ngpu, num_gpus, cpu_enabled;
	float similarity, sigma, sigma_step;
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	struct timespec tspec_tbefore, tspec_tafter;

	threshold_cv = 0;
	omp_set_nested(1);

	matrix = load_matrix(argv[1], &w, &h);
	ground_truth = load_matrix(argv[2], &w, &h);

	for(int i=0; i<w*h; i++)
		if(ground_truth[i] != 0 && ground_truth[i] != 1)
		{
			printf("This ground truth isn't binary. Exiting...");
			return(-1);
		}

	ncpu = atoi(argv[3]);
	ngpu = atoi(argv[4]);

	cpu_enabled = (ncpu > 0) ? 1: 0;
	omp_threads = ncpu;

	/* cpu memory allocation */
	noisy_matrix = mmalloc(h, w);
	edge = mmalloc(h, w);

	/* gpu memory allocation */
	dev_noisy_matrices = (int**)malloc(ngpu * sizeof(int*));
	dev_edges = (int**)malloc(ngpu * sizeof(int*));

	num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);

	dev_matrix = (int**)malloc(num_gpus*sizeof(int*));
	dev_ground_truth = (int**)malloc(num_gpus*sizeof(int*));
	states = (curandState_t**)malloc(num_gpus*sizeof(curandState_t*));

	size = h*w*sizeof(int);
	for(int i=0; i<num_gpus; i++)
	{
		cudaSetDevice(i);
		cudaMalloc(&dev_matrix[i], size);
		cudaMalloc(&dev_ground_truth[i], size);
		cudaMalloc(&states[i], h*w*sizeof(curandState_t));

		cudaMemcpy(dev_matrix[i], matrix, size, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_ground_truth[i], ground_truth, size, cudaMemcpyHostToDevice);
	}
	for(int i=0; i<ngpu; i++)
	{
		cudaSetDevice(i % num_gpus);
		cudaMalloc(&dev_noisy_matrices[i], size);
		cudaMalloc(&dev_edges[i], size);
	}


	exec_list = (execnode*)malloc(STEPS*REPS*sizeof(execnode));
	sigma_step = (SMAX-SMIN)/STEPS;
	sigma = SMIN;
	for(int i=0; i<STEPS; i++, sigma += sigma_step)
		for(int j=0; j<REPS; j++)
		{
			exec_list[i*REPS + j].sigma = sigma;
			exec_list[i*REPS + j].rep = j+1;
		}


	for(int i=0; i<num_gpus; i++)
	{
		cudaSetDevice(i);
		gpu_noise_init<<<BLOCKS, THREADS>>>(time(0), states[i], h*w);
		cudaDeviceSynchronize();
	}

	clock_gettime(CLOCK_MONOTONIC, &tspec_tbefore);

	#pragma omp parallel for private(similarity) shared(threshold_cv) schedule(dynamic, 1) num_threads(cpu_enabled+ngpu)
		for(int i=0; i<STEPS*REPS; i++)
		{
			int thread = omp_get_thread_num();
			struct timespec tspec_before, tspec_after, before, after;

			clock_gettime(CLOCK_MONOTONIC, &tspec_before);

			if(thread < ngpu)
			{
				cudaSetDevice(thread % num_gpus);
				timeit_gpu_kernel(before, after, gpu_noise_maker, states[thread % num_gpus], dev_matrix[thread % num_gpus], dev_noisy_matrices[thread], 1.0, exec_list[i].sigma, h*w);
				exec_list[i].time_noise = 1000*time_diff(before, after);

				timeit_gpu_kernel(before, after, gpu_edge_detector_cv, dev_noisy_matrices[thread], dev_edges[thread], w, h);
				exec_list[i].time_edge = 1000*time_diff(before, after);

				timeit_gpu(before, after, gpu_find_threshold_optimized, 0, threshold_cv, 8, 2, 0.5, dev_edges[thread], dev_ground_truth[thread % num_gpus], w, h, gpu_edge_comparison, &threshold_cv, &similarity);
				exec_list[i].time_perf = 1000*time_diff(before, after);

				// gpu_noise_maker<<<BLOCKS, THREADS>>>(states[thread % num_gpus], dev_matrix[thread % num_gpus], dev_noisy_matrices[thread], 1.0, exec_list[i].sigma, h*w);
				// gpu_edge_detector_cv<<<BLOCKS, THREADS>>>(dev_noisy_matrices[thread], dev_edges[thread], w, h);
				// gpu_find_threshold_optimized(0, threshold_cv, 8, 2, 0.5, dev_edges[thread], dev_ground_truth[thread % num_gpus], w, h, gpu_edge_comparison, &threshold_cv, &similarity);
				cudaDeviceSynchronize();
				exec_list[i].use_gpu = true;
			}
			else
			{
				timeit(before, after, noise_maker_multiplicative, matrix, noisy_matrix, h, w, exec_list[i].sigma, tm.tm_sec*thread+tm.tm_mday*tm.tm_yday);
				exec_list[i].time_noise = 1000*time_diff(before, after);

				timeit(before, after, edge_detector_cv, noisy_matrix, edge, w, h);
				exec_list[i].time_edge = 1000*time_diff(before, after);

				timeit(before, after, find_threshold_optimized, 0, threshold_cv, 8, 2, 0.5, edge, ground_truth, w, h, edge_comparison, &threshold_cv, &similarity);
				exec_list[i].time_perf = 1000*time_diff(before, after);

				// noise_maker_multiplicative(matrix, noisy_matrix, h, w, exec_list[i].sigma, tm.tm_sec*thread+tm.tm_mday*tm.tm_yday);
				// edge_detector_cv(noisy_matrix, edge, w, h);
				// find_threshold_optimized(0, threshold_cv, 8, 2, 0.5, edge, ground_truth, w, h, edge_comparison, &threshold_cv, &similarity);
				exec_list[i].use_gpu = false;
			}
			// printf("%d %.3f %2d %.6f\n", thread, exec_list[i].sigma, exec_list[i].rep, similarity);

			clock_gettime(CLOCK_MONOTONIC, &tspec_after);
			exec_list[i].time_ms = 1000*time_diff(tspec_before, tspec_after);
		}

	for(int i=0; i<num_gpus; i++)
		cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &tspec_tafter);

	int gpu_count = 0;
	for(int i=0; i<STEPS*REPS; i++)
	{
		printf("%.3f %2d %d %.3f %.3f %.3f %.3f\n", exec_list[i].sigma, exec_list[i].rep, exec_list[i].use_gpu, exec_list[i].time_noise, exec_list[i].time_edge, exec_list[i].time_perf, exec_list[i].time_ms);
		gpu_count += exec_list[i].use_gpu;
	}
	printf("cpu: %d gpu: %d\n", STEPS*REPS - gpu_count, gpu_count);

	printf("%s %d %d %.3f\n", name(argv[1]), ncpu, ngpu, time_diff(tspec_tbefore, tspec_tafter));


	/* be free memory!!! */
	mfree(matrix);
	mfree(ground_truth);
	mfree(noisy_matrix);
	mfree(edge);
	for(int i=0; i<num_gpus; i++)
	{
		cudaFree(dev_matrix[i]);
		cudaFree(dev_ground_truth[i]);
		cudaFree(states[i]);
	}
	free(dev_matrix);
	free(dev_ground_truth);
	free(states);
	for(int i=0; i<ngpu; i++)
	{
		cudaFree(dev_noisy_matrices[i]);
		cudaFree(dev_edges[i]);
	}
	free(dev_noisy_matrices);
	free(dev_edges);

	free(exec_list);


	cudaDeviceReset();
	return 0;
}

void usage()
{
	printf(
		"Usage:\n"
		);
}

char* name(char* path)
{
    int i;
    char* base;

    base = basename(path);

    for(i=0;base[i]!='.' && base[i]!='\0';i++);
    base[i] = '\0';

    return base;
}

float time_diff(struct timespec before, struct timespec after)
{
	return (after.tv_sec - before.tv_sec) + (after.tv_nsec - before.tv_nsec) / BILLION;
}

/*
	----
	./main <matrix_file> <ground_truth_file> <ncpu> <ngpu>
	----
*/