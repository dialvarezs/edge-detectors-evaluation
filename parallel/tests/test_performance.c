#include <stdio.h>
#include <sys/time.h>
#include <getopt.h>
#include <libgen.h>
#include "../utils/matrix_ops.h"
#include "../performance/performance.h"

void usage();

int main(int argc, char** argv )
{
 	int* edge = NULL;
 	int* ground_truth = NULL;
 	int c, fflag, aflag, m;
 	int w, h;
 	int threshold;
 	float similarity, secs, secs_load;
 	char * edge_fn, * gt_fn, * out_fn;
 	struct timeval tval_before, tval_after;
 	FILE* outfile;

 	fflag = aflag = 0;
	while((c = getopt(argc, argv, "f:ah")) != -1)
	{
		switch (c)
		{
			case 'f':
				fflag = 1;
				out_fn = optarg;
				break;
			case 'a':
				aflag = 1;
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

	//data load
	gettimeofday(&tval_before, NULL);

	edge = load_matrix(edge_fn, &w, &h);
	ground_truth = load_matrix(gt_fn, &w, &h);

	gettimeofday(&tval_after, NULL);
	secs_load = (float)(tval_after.tv_usec - tval_before.tv_usec) / 1000000 + (float)(tval_after.tv_sec - tval_before.tv_sec);

	//computing
 	gettimeofday(&tval_before, NULL);

 	if(m == 'e')
 		find_threshold_exhaustive(edge, ground_truth, w, h, edge_comparison, &threshold, &similarity);
 	else
 		find_threshold_optimized(0, 0, 8, 2, 0.5, edge, ground_truth, w, h, edge_comparison, &threshold, &similarity);

 	gettimeofday(&tval_after, NULL);
 	secs = (float)(tval_after.tv_usec - tval_before.tv_usec) / 1000000 + (float)(tval_after.tv_sec - tval_before.tv_sec);


 	fprintf(outfile, "%s %c %d %d %f %f %f\n", basename(edge_fn), m, w*h, threshold, similarity, secs, secs_load);

 	fclose(outfile);

 	mfree(edge);
 	mfree(ground_truth);

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
		"\t-a: prints all the calculated values, not only the optimal(only applies to exhaustive method)\n"
		"Output:\n"
		"\t<edge_file> <method> <matrix_size> <threshold> <value> <compute_time> <read_time>\n"
		);
}
