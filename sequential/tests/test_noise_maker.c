#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../utils/matrix_ops.h"

void usage();
float time_diff(struct timeval before, struct timeval after);

void usage();

int main(int argc, char** argv)
{
	int* matrix;
	int* noisy_matrix;
	int h, w;
	float sigma;
	struct timeval tval_before, tval_after;
	
	if(argc < 4)
	{
		usage();
		return -1;
	}

	matrix = load_matrix(argv[1], &w, &h);
	sigma = atof(argv[3]);

	noisy_matrix = malloc(w*h*sizeof(int));

	gettimeofday(&tval_before, NULL);
	noise_maker_multiplicative(matrix, noisy_matrix, h, w, sigma);
	//noise_maker_additive(matrix, noisy_matrix, h, w, sigma);
	//noise_maker_saltpepper(matrix, noisy_matrix, h, w, sigma);
	gettimeofday(&tval_after, NULL);
	printf("%.3f", 1000*time_diff(tval_before, tval_after));

	save_matrix(argv[2], noisy_matrix, w, h);

	mfree(matrix);
	mfree(noisy_matrix);

	return 0;
}

void usage()
{
	printf(
		"Usage:\n"
		"\t./noise_maker <matrix_file> <noisy_matrix_file(out)> <sigma>\n"
		);
}

float time_diff(struct timeval before, struct timeval after)
{
	return (float)(after.tv_usec - before.tv_usec) / 1000000 + (float)(after.tv_sec - before.tv_sec);
}
