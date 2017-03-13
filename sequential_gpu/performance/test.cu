#include <stdio.h>
#include <getopt.h>
#include <cuda.h>
#include "../utils/matrix_ops.h"
#include "performance_gpu.h"

void usage();

int main(int argc, char** argv )
{
 	int* edge = NULL;
 	int* ground_truth = NULL;
 	int* dev_edge = NULL;
 	int* dev_ground_truth = NULL;
 	int c, fflag, m;
 	int w, h, size;
 	int threshold;
 	float similarity, secs, secs_load;
 	char* edge_fn, * gt_fn, * out_fn;
 	FILE* outfile;
 	cudaEvent_t start, stop;
 	

 	fflag = 0;
	while((c = getopt(argc, argv, "f:ah")) != -1)
	{
		switch (c)
		{
			case 'f':
				fflag = 1;
				out_fn = optarg;
				break;
			default:
				break;
		}
	}
	if(optopt != 0 || argc-optind < 3)
	{
		usage();
		return -1;
	}
	else
	{
		edge_fn = argv[optind];
		gt_fn = argv[++optind];
		m = argv[++optind][0];
		if(m != 'e' && m!= 'o')
		{
			fprintf(stderr, "Method not valid\n");
			usage();
			return -1;
		}
	}

	if(fflag)
		outfile = fopen(out_fn, "w");
	else
		outfile = stdout;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//data load
	cudaEventRecord(start, 0);

	edge = load_matrix(edge_fn, &w, &h);
	ground_truth = load_matrix(gt_fn, &w, &h);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&secs_load, start, stop);

	size = h*w*sizeof(int);

	cudaMalloc(&dev_edge, size);
	cudaMalloc(&dev_ground_truth, size);

	cudaMemcpy(dev_edge, edge, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ground_truth, ground_truth, size, cudaMemcpyHostToDevice);

	//computing
 	cudaEventRecord(start, 0);

	if(m == 'e')
		gpu_find_threshold_exhaustive(dev_edge, dev_ground_truth, w, h, gpu_edge_comparison, &threshold, &similarity);
	else
		gpu_find_threshold_optimized(0, 0, 8, 2, 0.5, dev_edge, dev_ground_truth, w, h, gpu_edge_comparison, &threshold, &similarity);

 	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&secs, start, stop);


 	fprintf(outfile, "%s %c %d %d %f %f %f\n", basename(edge_fn), m, w*h, threshold, similarity, secs, secs_load);

 	fclose(outfile);

 	mfree(edge);
 	mfree(ground_truth);
 	cudaFree(dev_edge);
 	cudaFree(dev_ground_truth);


 	cudaDeviceReset();
 	return 0;
}

void usage()
{
	printf(
		"Usage:\n"
		"\t./perfomance <edge_file> <ground_truth_file> <method> <options>\n"
		"\tmethod: e(exhaustive) or o(optimized)\n"
		"Options:\n"
		"\t-f <filename>: writes output to file specified (by default prints to stdout)\n"
		"Output:\n"
		"\t<edge_file> <method> <matrix_size> <threshold> <value> <compute_time> <read_time>\n"
		);
}