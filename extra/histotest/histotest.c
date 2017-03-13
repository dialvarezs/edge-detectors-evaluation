#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double gaussrand(float sigma, float mu)
{
	static double V1, V2, S;
	static int phase = 0;
	double U1, U2;
	double X;

	if(phase == 0) {
		do {
			U1 = (double)rand() / RAND_MAX;
			U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
			} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return mu + X * sigma;
}

int main()
{
	for(int i=0; i<1000000; i++)
		printf("%lf\n", gaussrand(0.2, 1));
	return 0;
}
