#include <stdio.h>
#include "../utils/matrix_ops.h"
#include "../utils/gpu_consts.h"
#include "../edge_detector/edge_detectors_gpu.h"

void usage();

int main(int argc, char** argv)
{
	int* mat = NULL;
	int* mask = NULL;
	int* edge = NULL;
	int* dev_mat = NULL;
	int* dev_mask = NULL;
	int* dev_edge = NULL;
	int w, h, size;
	char detector;
	float secs, secs_load, secs_save;
	cudaEvent_t start, stop;


	if(argc < 4 || (argv[3][0] == 'g' && argc < 5))
	{
		usage();
		return -1;
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//data load
	cudaEventRecord(start, 0);

	mat = load_matrix(argv[1], &w, &h);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&secs_load, start, stop);

	detector = argv[3][0];
	edge = mmalloc(h, w);

	size = h*w*sizeof(int);
	cudaMalloc(&dev_mat, size);
	cudaMalloc(&dev_edge, size);
	cudaMalloc(&dev_mask, 9*sizeof(int));

	cudaMemcpy(dev_mat, mat, size, cudaMemcpyHostToDevice);

	//computing
	cudaEventRecord(start, 0);

	if(detector == 'g')
	{
		mask = load_mask(argv[4]);
		cudaMemcpy(dev_mask, mask, 9*sizeof(int), cudaMemcpyHostToDevice);

		gpu_edge_detector_g<<<BLOCKS, THREADS>>>(dev_mat, dev_edge, w, h, dev_mask);
	}
	else if(detector == 'c')
	{
		gpu_edge_detector_cv<<<BLOCKS, THREADS>>>(dev_mat, dev_edge, w, h);
	}
	else
	{
		fprintf(stderr, "Detector not valid\n");
		usage();
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&secs, start, stop);

	cudaMemcpy(edge, dev_edge, size, cudaMemcpyDeviceToHost);

 	//data save
 	cudaEventRecord(start, 0);
	
	save_matrix(argv[2], edge, w, h);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&secs_save, start, stop);

	printf("%s %c %d %f %f %f\n", basename(argv[1]), detector, w*h, secs, secs_load, secs_save);

	mfree(mat);
	mfree(mask);
	mfree(edge);
	cudaFree(dev_mat);
	cudaFree(dev_mask);
	cudaFree(dev_edge);

	return 0;
}

void usage()
{
	printf(
		"Usage:\n"
		"\t./edge_detector <matrix_file> <edge_out_file> g <mask_file> (gradient detector)\n"
		"\t./edge_detector <matrix_file> <edge_out_file> c (coefficient of variation detector)\n"
		"Output:\n"
		"\t<matrix_file> <edge_detector> <size> <compute_time> <read_time> <write_time>\n"
		);
}