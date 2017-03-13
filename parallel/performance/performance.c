#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../utils/matrix_ops.h"
#include "performance.h"

void find_threshold_exhaustive(int* edge, int* ground_truth, int width, int height, comparison_index comparison, int* threshold, float* similarity)
{
	int* hist = NULL;
	float value_max, value_tmp;
	int t;

	hist = histogram(edge, width, height);

	value_max = 0.0;
	t = 0;
	for(int i=0; i<256; i++)
		if(hist[i] > 0)
		{
			value_tmp = edge_comparison(edge, ground_truth, i, width, height);
			// printf("i=%d max=%f tmp=%f\n", i, value_max, value_tmp);
			if(value_tmp > value_max)
			{
				value_max = value_tmp;
				t = i;
			}
		}

	free(hist);

	*threshold = t;
	*similarity = value_max;
}

/* Davies, Swann and Campey */
void find_threshold_optimized(int iteration, int initval, int delta, int tolerance, float K,
	int* edge, int* ground_truth, int width, int height, comparison_index comparison, int* threshold, float* similarity)
{
	int i, j, dir, t;
	float t_value;
	int x[20]; //have the points of the evaluations
	float y[20]; //the values calculated

	dir = 0; // -1=left, 1=right
	for(i=1, j=0; i<3; i++, j++)
	{
		x[i] = initval + delta*j;
		y[i] = edge_comparison(edge, ground_truth, x[i], width, height);
	}
	//here i=3 and j=2

	if(y[2] > y[1])
		dir = 1;
	else
	{
		x[0] = x[2];
		y[0] = y[2];
		x[2] = initval - delta;
		y[2] = edge_comparison(edge, ground_truth, x[2], width, height);
		
		if(y[2] > y[1])
			dir = -1;
		else
		{
			i = 1;
			j = 1;
		}
	}

	if(dir != 0)
	{
		for(; ; i++, j*=2)
		{
			x[i] = x[i-1] + (dir*j*delta);
			if(x[i]<0 || x[i]>255)
			{
				i--;
				break;
			}
			y[i] = edge_comparison(edge, ground_truth, x[i], width, height);
			if(y[i] < y[i-1])
				break;
		}

		x[i+1] = x[i];
		y[i+1] = y[i];
		j/=2;
		x[i] = x[i-1] + (dir*j*delta);
		y[i] = edge_comparison(edge, ground_truth, x[i], width, height);

		if(y[i] < y[i-1])
			i--;
	}

	t = (x[i] + (delta*(y[i-1]-y[i+1]))/(2*(y[i+1]-2*y[i]+y[i-1]))) + 0.5; //0.5=approximation
	t_value = edge_comparison(edge, ground_truth, t, width, height);
	
	// printf("iteration %d [delta:%d tolerance:%d]\n", iteration, delta*j, tolerance);
	// for(i=0; i<10; i++)
	// 	printf("%2d x:%d y:%f\n", i, x[i], y[i]);

	if(delta*j < tolerance)
	{
		*threshold = t;
		*similarity = t_value;
		return;
	}
	else
		return find_threshold_optimized(iteration+1, t, delta*K+0.5, tolerance, K,
			edge, ground_truth, width, height, comparison, threshold, similarity);
}

float edge_comparison(int* edge, int* ground_truth, int threshold, int width, int height)
{
	int success, false_pos, false_neg;

	success = false_neg = false_pos = 0;
	#pragma omp parallel for schedule(dynamic,8) reduction(+:success,false_pos,false_neg)
		for(int i=0; i<height; i++)
			for(int j=0; j<width; j++)
			{
				if(edge[i*width + j] >= threshold && ground_truth[i*width + j] == 1)
					success++;
				else if(edge[i*width + j] >= threshold && ground_truth[i*width + j] == 0)
					false_pos++;
				else if(edge[i*width + j] < threshold && ground_truth[i*width + j] == 1)
					false_neg++;
			}

	// printf("succ:%d fp:%d; fn:%d = %f\n", success, false_pos, false_neg, (float)success/(success + false_neg + false_pos));

	if(success == 0) //avoid 0/0
		return 0.0;
	return (float)success/(success + false_neg + false_pos);
}

int* binarization(int* edge, int* binarized, int width, int height, int threshold)
{
	// int** binarized;

	// binarized = mmalloc(height, width); 

	#pragma omp parallel for schedule(dynamic,8)
		for(int i=0; i<height; i++)
			for(int j=0; j<width; j++)
			{
				if(edge[i*width + j] >= threshold) //the edges are highlighted with lighter shades
					binarized[i*width + j] = 1;
				else
					binarized[i*width + j] = 0;
			}

	return binarized;
}

int* histogram(int* matrix, int width, int height)
{
	int* hist = NULL;

	hist = malloc(256*sizeof(int));

	for(int i=0; i<256; i++)
		hist[i] = 0;

	for(int i=0; i<height; i++)
		for(int j=0; j<width; j++)
			hist[matrix[i*width + j]]++;

	// for(int i=0; i<255; i++)
	// 	printf("hist(%d): %d\n", i, hist[i]);

	return hist;
}
