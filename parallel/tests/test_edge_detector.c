#include <stdio.h>
#include <sys/time.h>
#include <libgen.h>
#include "../utils/matrix_ops.h"
#include "../edge_detector/edge_detectors.h"

void usage();

int main(int argc, char** argv)
{
	int* mat = NULL;
	int* mask = NULL;
	int* edge = NULL;
	int w,h;
	char detector;
	struct timeval tval_before, tval_after;
	float secs, secs_load, secs_save;

	if(argc < 4 || (argv[3][0] == 'g' && argc < 5))
	{
		usage();
		return -1;
	}

	//data load
	gettimeofday(&tval_before, NULL);

	mat = load_matrix(argv[1], &w, &h);
	
	gettimeofday(&tval_after, NULL);
 	secs_load = (float)(tval_after.tv_usec - tval_before.tv_usec) / 1000000 + (float)(tval_after.tv_sec - tval_before.tv_sec);

	detector = argv[3][0];
	edge = mmalloc(h, w);

	//computing
	gettimeofday(&tval_before, NULL);

	if(detector == 'g')
	{
		mask = load_mask(argv[4]);
		edge_detector_g(mat, edge, w, h, mask);
	}
	else if(detector == 'c')
		edge_detector_cv(mat, edge, w, h);
	else
	{
		fprintf(stderr, "Detector not valid\n");
		usage();
	}

	gettimeofday(&tval_after, NULL);
 	secs = (float)(tval_after.tv_usec - tval_before.tv_usec) / 1000000 + (float)(tval_after.tv_sec - tval_before.tv_sec);

 	//data save
 	gettimeofday(&tval_before, NULL);
	
	save_matrix(argv[2], edge, w, h);
	
	gettimeofday(&tval_after, NULL);
 	secs_save = (float)(tval_after.tv_usec - tval_before.tv_usec) / 1000000 + (float)(tval_after.tv_sec - tval_before.tv_sec);

	printf("%s %c %d %f %f %f\n", basename(argv[1]), detector, w*h, secs, secs_load, secs_save);

	mfree(mat);
	mfree(mask);
	mfree(edge);

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
