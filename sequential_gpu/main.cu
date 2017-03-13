#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <libgen.h>
#include <sys/stat.h>
#include <cuda.h>
#include "utils/matrix_ops.h"
#include "utils/gpu_consts.h"
#include "edge_detector/edge_detectors_gpu.h"
#include "performance/performance_gpu.h"

void usage();
char* name(char* path);

int main(int argc, char** argv)
{
	int* matrix = NULL;
	int* ground_truth = NULL;
	int* edge_binarized = NULL;
	int* mask = NULL;
	int* dev_matrix = NULL;
	int* dev_noisy_matrix = NULL;
	int* dev_ground_truth = NULL;
	int* dev_edge = NULL;
	int* dev_edge_binarized = NULL;
	int* dev_mask = NULL;
	curandState_t* states = NULL;
	FILE* fresults = NULL;
	FILE* ftimes = NULL;

	int w, h, size, steps, reps, threshold, threshold_cv, threshold_g;
	float sigma, sigma_step, sigma_min, sigma_max, similarity, msecs;
	char edge_dec, perf_fn, save_edge, dir[50], namebuffer[100], buffer[100];
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	cudaEvent_t start, stop, tstart, tstop;

	threshold_cv=threshold_g=0;

	matrix = load_matrix(argv[1], &w, &h);
	ground_truth = load_matrix(argv[2], &w, &h);

	sigma_min = atof(argv[4]);
	sigma_max = atof(argv[5]);
	steps = atoi(argv[6]);
	reps = atoi(argv[7]);
	edge_dec = argv[8][0];
	save_edge = argv[9][0];
	perf_fn = argv[10][0];

	if(edge_dec == 'g' || edge_dec == 'a')
	{
		mask = load_mask(argv[11]);
		cudaMalloc(&dev_mask, 9*sizeof(int));
		cudaMemcpy(dev_mask, mask, 9*sizeof(int), cudaMemcpyHostToDevice);
	}

	sprintf(dir, "%s/exec_gpu_%s_%d%02d%02d-%02d%02d%02d", argv[3], name(argv[1]), tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
	mkdir(dir, 0755);
	if(save_edge == 'y')
	{
		sprintf(namebuffer, "%s/edges", dir);
		mkdir(namebuffer, 0755);
	}

	sprintf(namebuffer, "%s/results.dat", dir);
	fresults = fopen(namebuffer, "w");

	sprintf(namebuffer, "%s/times.dat", dir);
	ftimes = fopen(namebuffer, "w");

	size = h*w*sizeof(int);
	edge_binarized = (int*)mmalloc(h, w);
	cudaMalloc(&dev_matrix, size);
	cudaMalloc(&dev_noisy_matrix, size);
	cudaMalloc(&dev_ground_truth, size);
	cudaMalloc(&dev_edge, size);
	cudaMalloc(&states, h*w*sizeof(curandState_t));
	if(save_edge == 'y')
		cudaMalloc(&dev_edge_binarized, size);

	cudaMemcpy(dev_matrix, matrix, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ground_truth, ground_truth, size, cudaMemcpyHostToDevice);


	printf("exec_gpu_%s_%d%02d%02d-%02d%02d%02d\n", name(argv[1]), tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&tstart);
	cudaEventCreate(&tstop);

	sigma_step = (sigma_max-sigma_min)/steps;
	sigma=sigma_min;
	gpu_noise_init<<<BLOCKS, THREADS>>>(time(0), states, h*w);
	for(int i=0; i<steps; i++)
	{
		for(int j=0; j<reps; j++)
		{
			sprintf(buffer, "%.3f %d", sigma, j+1);

			cudaEventRecord(tstart, 0);

			cudaEventRecord(start, 0);
			gpu_noise_maker<<<BLOCKS, THREADS>>>(states, dev_matrix, dev_noisy_matrix, 1.0, sigma, h*w);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&msecs, start, stop);

			sprintf(buffer, "%s %.3f", buffer, msecs);

			if(edge_dec=='c' || edge_dec=='a')
			{
				cudaEventRecord(start, 0);
				gpu_edge_detector_cv<<<BLOCKS, THREADS>>>(dev_noisy_matrix, dev_edge, w, h);
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&msecs, start, stop);

				sprintf(buffer, "%s %.3f", buffer, msecs);

				if(perf_fn=='o' || perf_fn=='a')
				{
					cudaEventRecord(start, 0);
					gpu_find_threshold_optimized(0, threshold_cv, 8, 2, 0.5, dev_edge, dev_ground_truth, w, h, gpu_edge_comparison, &threshold_cv, &similarity);
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&msecs, start, stop);

					sprintf(buffer, "%s %.3f", buffer, msecs);

					fprintf(fresults, "%.3f %d cv opt %d %.6f\n", sigma, j+1, threshold_cv, similarity);

					sprintf(namebuffer, "%s/edges/%.3f_%d_cv_opt.dat", dir, sigma, j+1);
					if(save_edge == 'y')
					{
						gpu_binarization<<<BLOCKS, THREADS>>>(dev_edge, dev_edge_binarized, w, h, threshold_cv);
						cudaMemcpy(edge_binarized, dev_edge_binarized, size, cudaMemcpyDeviceToHost);
						save_matrix(namebuffer, edge_binarized, w, h);
					}
				}
				if(perf_fn=='e' || perf_fn=='a')
				{
					cudaEventRecord(start, 0);
					gpu_find_threshold_exhaustive(dev_edge, dev_ground_truth, w, h, gpu_edge_comparison, &threshold, &similarity);
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&msecs, start, stop);

					sprintf(buffer, "%s %.3f", buffer, msecs);

					fprintf(fresults, "%.3f %d cv exh %d %.6f\n", sigma, j+1, threshold, similarity);

					sprintf(namebuffer, "%s/edges/%.3f_%d_cv_exh.dat", dir, sigma, j+1);
					if(save_edge == 'y')
					{
						gpu_binarization<<<BLOCKS, THREADS>>>(dev_edge, dev_edge_binarized, w, h, threshold);
						cudaMemcpy(edge_binarized, dev_edge_binarized, size, cudaMemcpyDeviceToHost);
						save_matrix(namebuffer, edge_binarized, w, h);
					}
				}
			}
			if(edge_dec=='g' || edge_dec=='a')
			{
				cudaEventRecord(start, 0);
				gpu_edge_detector_g<<<BLOCKS, THREADS>>>(dev_noisy_matrix, dev_edge, w, h, dev_mask);
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&msecs, start, stop);

				sprintf(buffer, "%s %.3f", buffer, msecs);

				if(perf_fn=='o' || perf_fn=='a')
				{
					cudaEventRecord(start, 0);
					gpu_find_threshold_optimized(0, threshold_g, 8, 2, 0.5, dev_edge, dev_ground_truth, w, h, gpu_edge_comparison, &threshold_g, &similarity);
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&msecs, start, stop);

					sprintf(buffer, "%s %.3f", buffer, msecs);

					fprintf(fresults, "%.3f %d g opt %d %.6f\n", sigma, j+1, threshold_g, similarity);

					sprintf(namebuffer, "%s/edges/%.3f_%d_g_opt.dat", dir, sigma, j+1);
					if(save_edge == 'y')
					{
						gpu_binarization<<<BLOCKS, THREADS>>>(dev_edge, dev_edge_binarized, w, h, threshold_g);
						cudaMemcpy(edge_binarized, dev_edge_binarized, size, cudaMemcpyDeviceToHost);
						save_matrix(namebuffer, edge_binarized, w, h);
					}
				}
				if(perf_fn=='e' || perf_fn=='a')
				{
					cudaEventRecord(start, 0);
					gpu_find_threshold_exhaustive(dev_edge, dev_ground_truth, w, h, gpu_edge_comparison, &threshold, &similarity);
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&msecs, start, stop);

					sprintf(buffer, "%s %.3f", buffer, msecs);

					fprintf(fresults, "%.3f %d g exh %d %.6f\n", sigma, j+1, threshold, similarity);

					sprintf(namebuffer, "%s/edges/%.3f_%d_g_exh.dat", dir, sigma, j+1);
					if(save_edge == 'y')
					{
						gpu_binarization<<<BLOCKS, THREADS>>>(dev_edge, dev_edge_binarized, w, h, threshold);
						cudaMemcpy(edge_binarized, dev_edge_binarized, size, cudaMemcpyDeviceToHost);
						save_matrix(namebuffer, edge_binarized, w, h);
					}
				}
			}
			cudaEventRecord(tstop, 0);
			cudaEventSynchronize(tstop);
			cudaEventElapsedTime(&msecs, tstart, tstop);
			sprintf(buffer, "%s %.3f", buffer, msecs);

			printf("%s\n", buffer);
			fprintf(ftimes, "%s\n", buffer);
		}
		sigma += sigma_step;
	}

	// printf("w:%d h:%d int:%ld\n%ld %ld %ld\n", w, h, sizeof(int*), malloc_usable_size(noisy_matrix), malloc_usable_size(edge), malloc_usable_size(edge_binarized));

	fclose(fresults);
	fclose(ftimes);

	mfree(matrix);
	mfree(ground_truth);
	mfree(edge_binarized);
	mfree(mask);
	cudaFree(dev_matrix);
	cudaFree(dev_noisy_matrix);
	cudaFree(dev_ground_truth);
	cudaFree(dev_edge);
	cudaFree(dev_edge_binarized);
	cudaFree(dev_mask);


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

/*
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