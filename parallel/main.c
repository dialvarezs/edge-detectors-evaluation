#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <libgen.h>
#include <sys/time.h>
#include <sys/stat.h>
#include "matrix_ops/matrix_ops.h"
#include "edge_detector/edge_detectors.h"
#include "performance/performance.h"


void usage();
float time_diff(struct timeval before, struct timeval after);
char* name(char* path);

int main(int argc, char** argv)
{
	int* matrix = NULL;
	int* noisy_matrix = NULL;
	int* ground_truth = NULL;
	int* edge = NULL;
	int* edge_binarized = NULL;
	int* mask = NULL;
	int w, h, steps, reps, threshold, threshold_cv, threshold_g;
	float sigma, sigma_step, sigma_min, sigma_max, similarity;
	char edge_dec, perf_fn, save_edge, dir[50], namebuffer[100], buffer[100];
	int i, j;
	FILE* fresults = NULL;
	FILE* ftimes = NULL;
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	struct timeval tval_before, tval_after, tval_tbefore, tval_tafter;

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
		mask = load_mask(argv[11]);

	sprintf(dir, "%s/exec_par_%s_%d%02d%02d-%02d%02d%02d", argv[3], name(argv[1]), tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
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

	noisy_matrix = mmalloc(h, w);
	edge = mmalloc(h, w);
	if(save_edge == 'y')
		edge_binarized = mmalloc(h, w);

	printf("exec_par_%s_%d%02d%02d-%02d%02d%02d\n", name(argv[1]), tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

	sigma_step = (sigma_max-sigma_min)/steps;
	for(i=0, sigma=sigma_min; i<steps; i++)
	{
		for(j=0; j<reps; j++)
		{
			sprintf(buffer, "%.3f %d", sigma, j+1);

			gettimeofday(&tval_tbefore, NULL);

			gettimeofday(&tval_before, NULL);
			noisy_matrix = noise_maker(matrix, noisy_matrix, h, w, sigma);
			gettimeofday(&tval_after, NULL);
			sprintf(buffer, "%s %.3f", buffer, 1000*time_diff(tval_before, tval_after));

			if(edge_dec=='c' || edge_dec=='a')
			{
				gettimeofday(&tval_before, NULL);
				edge = edge_detector_cv(noisy_matrix, edge, w, h);
				gettimeofday(&tval_after, NULL);
				sprintf(buffer, "%s %.3f", buffer, 1000*time_diff(tval_before, tval_after));

				if(perf_fn=='o' || perf_fn=='a')
				{
					gettimeofday(&tval_before, NULL);
					find_threshold_optimized(0, threshold_cv, 8, 2, 0.5, edge, ground_truth, w, h, edge_comparison, &threshold_cv, &similarity);
					gettimeofday(&tval_after, NULL);
					sprintf(buffer, "%s %.3f", buffer, 1000*time_diff(tval_before, tval_after));

					fprintf(fresults, "%.3f %d cv opt %d %.6f\n", sigma, j+1, threshold_cv, similarity);

					sprintf(namebuffer, "%s/edges/%.3f_%d_cv_opt.dat", dir, sigma, j+1);
					if(save_edge == 'y')
						save_matrix(namebuffer, binarization(edge, edge_binarized, w, h, threshold_cv), w, h);
				}
				if(perf_fn=='e' || perf_fn=='a')
				{
					gettimeofday(&tval_before, NULL);
					find_threshold_exhaustive(edge, ground_truth, w, h, edge_comparison, &threshold, &similarity);
					gettimeofday(&tval_after, NULL);
					sprintf(buffer, "%s %.3f", buffer, 1000*time_diff(tval_before, tval_after));

					fprintf(fresults, "%.3f %d cv exh %d %.6f\n", sigma, j+1, threshold, similarity);

					sprintf(namebuffer, "%s/edges/%.3f_%d_cv_exh.dat", dir, sigma, j+1);
					if(save_edge == 'y')
						save_matrix(namebuffer, binarization(edge, edge_binarized, w, h, threshold), w, h);
				}
			}
			if(edge_dec=='g' || edge_dec=='a')
			{
				gettimeofday(&tval_before, NULL);
				edge = edge_detector_g(noisy_matrix, edge, w, h, mask);
				gettimeofday(&tval_after, NULL);
				sprintf(buffer, "%s %.3f", buffer, 1000*time_diff(tval_before, tval_after));

				if(perf_fn=='o' || perf_fn=='a')
				{
					gettimeofday(&tval_before, NULL);
					find_threshold_optimized(0, threshold_g, 8, 2, 0.5, edge, ground_truth, w, h, edge_comparison, &threshold_g, &similarity);
					gettimeofday(&tval_after, NULL);
					sprintf(buffer, "%s %.3f", buffer, 1000*time_diff(tval_before, tval_after));

					fprintf(fresults, "%.3f %d g opt %d %.6f\n", sigma, j+1, threshold_g, similarity);

					sprintf(namebuffer, "%s/edges/%.3f_%d_g_opt.dat", dir, sigma, j+1);
					if(save_edge == 'y')
						save_matrix(namebuffer, binarization(edge, edge_binarized, w, h, threshold_g), w, h);
				}
				if(perf_fn=='e' || perf_fn=='a')
				{
					gettimeofday(&tval_before, NULL);
					find_threshold_exhaustive(edge, ground_truth, w, h, edge_comparison, &threshold, &similarity);
					gettimeofday(&tval_after, NULL);
					sprintf(buffer, "%s %.3f", buffer, 1000*time_diff(tval_before, tval_after));

					fprintf(fresults, "%.3f %d g exh %d %.6f\n", sigma, j+1, threshold, similarity);

					sprintf(namebuffer, "%s/edges/%.3f_%d_g_exh.dat", dir, sigma, j+1);
					if(save_edge == 'y')
						save_matrix(namebuffer, binarization(edge, edge_binarized, w, h, threshold), w, h);
				}
			}
			gettimeofday(&tval_tafter, NULL);
			sprintf(buffer, "%s %.3f", buffer, 1000*time_diff(tval_tbefore, tval_tafter));

			printf("%s\n", buffer);
			fprintf(ftimes, "%s\n", buffer);
		}
		sigma += sigma_step;
	}

	// printf("w:%d h:%d int:%ld\n%ld %ld %ld\n", w, h, sizeof(int*), malloc_usable_size(noisy_matrix), malloc_usable_size(edge), malloc_usable_size(edge_binarized));

	mfree(matrix);
	mfree(noisy_matrix);
	mfree(ground_truth);
	mfree(edge);
	mfree(edge_binarized);
	mfree(mask);

	fclose(fresults);
	fclose(ftimes);

	return 0;
}

void usage()
{
	printf(
		"Usage:\n"
		);
}

float time_diff(struct timeval before, struct timeval after)
{
	return (float)(after.tv_usec - before.tv_usec) / 1000000 + (float)(after.tv_sec - before.tv_sec);
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