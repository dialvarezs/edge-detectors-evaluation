#include <stdlib.h>
#include <math.h>
#include "../utils/matrix_ops.h"


int* edge_detector_g(int* img, int* edge, int width, int height, int* mask)
{
	#pragma omp parallel for schedule(static)
		for(int i=1; i<height-1; i++)
			for(int j=1; j<width-1; j++)
			{
				int gx=0, gy=0;
				for(int k=i-1,m=0; m<3; k++,m++)
					for(int l=j-1,n=0; n<3; l++,n++)
					{
						gx += img[k*width + l]*mask[m*3 + n];
						gy += img[k*width + l]*mask[n*3 + m];
					}

				edge[i*width + j] = sqrt(pow((float)gx/6,2) + pow((float)gy/6,2));
			}

	//fill borders with adyacent values
	fill_borders(edge, width, height);

	return edge;
}

int* edge_detector_cv(int* img, int* edge, int width, int height)
{
	#pragma omp parallel for schedule(static)
		for(int i=1; i<height-1; i++)
			for(int j=1; j<width-1; j++)
			{
				float avg=0, sum=0;
				for(int k=i-1; k<i+2; k++)
					for(int l=j-1; l<j+2; l++)
						avg += img[k*width + l];
				avg /= 9;

				//sum = 0;
				for(int k=i-1; k<i+2; k++)
					for(int l=j-1; l<j+2; l++)
						sum += pow(img[k*width + l]-avg,2);
				if(sum > 0)
				{
					edge[i*width + j] = 255*sqrt(sum/9)/fabs(avg);
					if(edge[i*width + j] > 255)
						edge[i*width + j] = 255; //normalize values greatear than 1
				}
				else
					edge[i*width + j] = 0; //avoid 0/0
			}

	//fill borders with adyacent values
	fill_borders(edge, width, height);

	return edge;
}
