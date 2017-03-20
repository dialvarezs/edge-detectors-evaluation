#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../utils/matrix_ops.cuh"
#include "edge_detectors.cuh"


void edge_detector_g(int* img, int* edge, int width, int height, int* mask)
{
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

			edge[i*width + j] = sqrt(pow((float)gx,2) + pow((float)gy,2));
			if(edge[i*width + j] > 255) //just in case
				edge[i*width + j] = 255;
		}

	//fill borders with adyacent values
	fill_borders(edge, width, height);
}

void edge_detector_cv(int* img, int* edge, int width, int height)
{
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
}

void edge_detector_mcv(int* img, int* edge, int width, int height)
{
	int* values = NULL;

	values = (int*)malloc(9*sizeof(int));

	for(int i=1; i<height-1; i++)
		for(int j=1; j<width-1; j++)
		{
			float sum = 0;
			int median;
			for(int k=i-1, m=0; k<i+2; k++, m++)
				for(int l=j-1, n=0; l<j+2; l++, n++)
					values[3*m + n] = img[k*width + l];
			median = median_qselect(values, 9);

			for(int k=i-1; k<i+2; k++)
				for(int l=j-1; l<j+2; l++)
					sum += pow(img[k*width + l]-median,2);
			if(sum > 0)
			{
				edge[i*width + j] = 255*sqrt(sum/9)/abs(median);
				if(edge[i*width + j] > 255)
					edge[i*width + j] = 255; //normalize values greatear than 1
			}
			else
				edge[i*width + j] = 0; //avoid 0/0
		}

	free(values);

	//fill borders with adyacent values
	fill_borders(edge, width, height);
}

void filter_med(int* img, int* img_filtered, int width, int height)
{
	int* values = NULL;

	values = (int*)malloc(9*sizeof(int));

	for(int i=1; i<height-1; i++)
		for(int j=1; j<width-1; j++)
		{
			for(int k=i-1, m=0; k<i+2; k++, m++)
				for(int l=j-1, n=0; l<j+2; l++, n++)
					values[3*m + n] = img[k*width + l];

			img_filtered[i*width + j] = median_qselect(values, 9);
		}

	free(values);

	//fill borders with adyacent values
	fill_borders(img_filtered, width, height);
}

void filter_avg(int* img, int* img_filtered, int width, int height)
{
	float avg;

	for(int i=1; i<height-1; i++)
		for(int j=1; j<width-1; j++)
		{
			avg = 0;
			for(int k=i-1; k<i+2; k++)
				for(int l=j-1; l<j+2; l++)
					avg += img[k*width + l];
			avg /= 9;

			img_filtered[i*width + j] = (int)round(avg);
		}

	//fill borders with adyacent values
	fill_borders(img_filtered, width, height);
}


__global__ void gpu_edge_detector_g(int* img, int* edge, int width, int height, int* mask)
{
	int k, l, m, n, gx, gy;

	for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<width*height; i += blockDim.x * gridDim.x)
	{
		k = i-width;
		if(i%width == 0) //left
			k += 1;
		else if(i%width == width-1) //right
			k -= 1;
		if(i/width == 0) //up
			k += width;
		else if(i/width == height-1) //bottom
			k -= width;

		gx = gy = 0;
		for(m=0; m<3; m++, k+=width)
			for(l=k-1, n=0; n<3; n++, l++)
			{
				gx += img[l]*mask[m*3 + n];
				gy += img[l]*mask[n*3 + m];
			}
		edge[i] = sqrtf(powf((float)gx,2) + powf((float)gy,2));
		if(edge[i] > 255)
			edge[i] =  255;
	}
}

__global__ void gpu_edge_detector_cv(int* img, int* edge, int width, int height)
{
	int k, l, m, n;
	float avg, sum;

	for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<width*height; i += blockDim.x * gridDim.x)
	{
		k = i-width;
		if(i%width == 0) //left
			k += 1;
		else if(i%width == width-1) //right
			k -= 1;
		if(i/width == 0) //up
			k += width;
		else if(i/width == height-1) //bottom
			k -= width;

		avg = sum = 0;
		for(m=0; m<3; m++)
			for(l=k+(m*width)-1, n=0; n<3; l++, n++)
				avg += img[l];
		avg /= 9;
		for(m=0; m<3; m++)
			for(l=k+(m*width)-1, n=0; n<3; l++, n++)
				sum += powf(img[l]-avg, 2);

		if(sum > 0)
		{
			edge[i] = 255*sqrtf(sum/9)/fabsf(avg);
			if(edge[i] > 255)
				edge[i] = 255;
		}
		else
			edge[i] = 0;
	}
}