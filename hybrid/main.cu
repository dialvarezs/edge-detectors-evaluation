#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <libgen.h>
#include <omp.h>
// #include <sys/stat.h>
#include <cuda.h>
// #include "utils/matrix_ops.h"
// #include "edge_detector/edge_detectors.h"
// #include "performance/performance.h"
#include "utils/matrix_ops_gpu.cuh"
#include "utils/gpu_consts.cuh"
#include "edge_detector/edge_detectors_gpu.cuh"
#include "performance/performance_gpu.cuh"


#define BILLION 1E9
// #define timeit_gpu(before, after, f, ...) {\
// 	cudaDeviceSynchronize();\
// 	clock_gettime(CLOCK_MONOTONIC, &before);\
// 	f(__VA_ARGS__);\
// 	cudaDeviceSynchronize();\
// 	clock_gettime(CLOCK_MONOTONIC, &after);\
// }
// #define timeit_gpu_kernel(before, after, k, ...) {\
// 	cudaDeviceSynchronize();\
// 	clock_gettime(CLOCK_MONOTONIC, &before);\
// 	k<<<BLOCKS, THREADS>>>(__VA_ARGS__);\
// 	cudaDeviceSynchronize();\
// 	clock_gettime(CLOCK_MONOTONIC, &after);\
// }

#define SMIN 0.0
#define SMAX 0.2
#define STEPS 10
#define REPS 30

struct execution
{
	float sigma;
	int rep;
};
typedef struct execution execnode;


void usage();
char* name(char* path);
float time_diff(struct timespec before, struct timespec after);

int main(int argc, char** argv)
{
	int* matrix = NULL;
	int* ground_truth = NULL;
	int** noisy_matrices = NULL;
	int** edges = NULL;
	int** dev_matrix = NULL;
	int** dev_ground_truth = NULL;
	int** dev_noisy_matrices = NULL;
	int** dev_edges = NULL;
	curandState_t** states = NULL;
	execnode* exec_list = NULL;

	int w, h, size, threshold_cv, ncpu, ngpu, num_gpus, thread;
	float similarity, sigma, sigma_step;
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	struct timespec tspec_tbefore, tspec_tafter;

	threshold_cv = 0;

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

	noisy_matrices = (int**)malloc(ncpu * sizeof(int*));
	edges = (int**)malloc(ncpu * sizeof(int*));

	for(int i=0; i<ncpu; i++)
	{
		noisy_matrices[i] = mmalloc(h, w);
		edges[i] = mmalloc(h, w);
	}


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
	}
	for(int i=0; i<ngpu; i++)
	{
		cudaSetDevice(i % num_gpus);
		cudaMalloc(&dev_noisy_matrices[i], size);
		cudaMalloc(&dev_edges[i], size);
	}

	for(int i=0; i<num_gpus; i++)
	{
		cudaMemcpy(dev_matrix[i], matrix, size, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_ground_truth[i], ground_truth, size, cudaMemcpyHostToDevice);
	}


	exec_list = (execnode*)malloc(STEPS*REPS*sizeof(execnode));
	sigma_step = (SMAX-SMIN)/STEPS;
	sigma = SMIN;
	for(int i=0; i<STEPS; i++, sigma += sigma_step)
		for(int j=0; j<REPS; j++)
		{
			exec_list[i*j + j].sigma = sigma;
			exec_list[i*j + j].rep = j;
		}


	for(int i=0; i<num_gpus; i++)
		gpu_noise_init<<<BLOCKS, THREADS>>>(time(0), states[i], h*w);

	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &tspec_tbefore);

	#pragma omp parallel for schedule(guided)
		for(int i=0; i<STEPS*REPS; i++)
		{
			thread = omp_get_thread_num();
			if(thread < ngpu)
			{
				cudaSetDevice(thread % num_gpus);
				gpu_noise_maker<<<BLOCKS, THREADS>>>(states[thread % num_gpus], dev_matrix[thread % num_gpus], dev_noisy_matrices[thread], 1.0, exec_list[i].sigma, h*w);
				gpu_edge_detector_cv<<<BLOCKS, THREADS>>>(dev_noisy_matrices[thread], dev_edges[thread], w, h);
				gpu_find_threshold_optimized(0, threshold_cv, 8, 2, 0.5, dev_edges[thread], dev_ground_truth[thread % num_gpus], w, h, gpu_edge_comparison, &threshold_cv, &similarity);
			}
			else
			{
				noise_maker_multiplicative(matrix, noisy_matrices[thread-ngpu], h, w, exec_list[i].sigma);
				edge_detector_cv(matrix, edges[thread-ngpu], w, h);
				gpu_find_threshold_optimized(0, threshold_cv, 8, 2, 0.5, edges[thread], ground_truth, w, h, gpu_edge_comparison, &threshold_cv, &similarity);
			}
		}

	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &tspec_tafter);

	printf("exec_gpu_%s_%d%02d%02d-%02d%02d%02d %.3f\n", name(argv[1]), tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, 1000*time_diff(tspec_tbefore, tspec_tafter));

	mfree(matrix);
	mfree(ground_truth);
	for(int i=0; i<ncpu; i++)
	{
		mfree(noisy_matrices[i]);
		mfree(edges[i]);
	}
	cudaFree(dev_matrix);
	cudaFree(dev_ground_truth);
	for(int i=0; i<ngpu; i++)
	{
		cudaFree(dev_noisy_matrices[i]);
		cudaFree(dev_edges[i]);
	}


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


	exec:
	./main <matrix_file> <ground_truth_file> <output_dir> <sigma_min> <sigma_max> <steps> <reps> <c/g/a> <save_edge(y/n)> <o/e/a> <mask_file>
	out:
	(results) <sigma> <rep> <edge_detector> <perf_fn> <threshold> <similarity>
	(times) <sigma> <rep> <noise_maker> <edge_detector> <perf_fn> ... <+edges> ...

	Parámetros:
	- Matriz de imagen original
	- Matrix de terreno de la verdad (la matriz debe ser de 0 y 1 sólamente)
	- Directorio donde se guardarán los archivos(tiempos, resultados y mejores contornos)
	- Parámetros de ruido(sigma_min, sigma_max, intervalos, repeticiones)
	- Detectores de ruido a usar(cv, g). Si g agregar máscara como último parámetro.
	- Funciones de performance a usar(o, e)

	Salida:
	- Mejor contorno por cada detector, por cada función de performance y por cada matriz de ruido generada (<=4 por matriz de ruido).
	- Archivo con tiempos para generación de ruido, detectores de contorno y performance
	- Tiempos totales


	Procedimiento:
	- Cargar matriz original y terreno de la verdad
	por cada valor de ruido:
		por cada repetición para valor de ruido:
			- Generar matriz ruidosa
			por cada detector seleccionado:
				- Obtener contorno
				por cada función de performance:
					- Comparar contorno con terreno de la verdad
					- Guardar a archivo el mejor contorno
			- Imprimir resultados y tiempos a archivos separados
			- Guardar matriz ruidosa
	- Sumar tiempos e imprimir a archivo(o stdout)

	TODO:
	Arreglar la fuga de memoria!
	
	guardar:
	- (mejores contornos) nombreimg_sigma_rep_detector_perf.dat
	- (tiempos)
*/