#include "performance_gpu.cuh"
#include "../utils/gpu_consts.cuh"


void gpu_find_threshold_exhaustive(int* edge, int* ground_truth, int width, int height, comparison_index comparison, int* threshold, float* similarity)
{
	int* hist = NULL;
	float value_max, value_tmp;
	int t;

	hist = gpu_histogram(edge, width, height);

	value_max = 0.0;
	t = 0;
	for(int i=0; i<255; i++)
		if(hist[i] > 0)
		{
			value_tmp = gpu_edge_comparison(edge, ground_truth, i, width, height);
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
void gpu_find_threshold_optimized(int iteration, int initval, int delta, int tolerance, float K,
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
		y[i] = gpu_edge_comparison(edge, ground_truth, x[i], width, height);
	}
	//here i=3 and j=2

	if(y[2] > y[1])
		dir = 1;
	else
	{
		x[0] = x[2];
		y[0] = y[2];
		x[2] = initval - delta;
		y[2] = gpu_edge_comparison(edge, ground_truth, x[2], width, height);
		
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
			y[i] = gpu_edge_comparison(edge, ground_truth, x[i], width, height);
			if(y[i] < y[i-1])
				break;
		}

		x[i+1] = x[i];
		y[i+1] = y[i];
		j/=2;
		x[i] = x[i-1] + (dir*j*delta);
		y[i] = gpu_edge_comparison(edge, ground_truth, x[i], width, height);

		if(y[i] < y[i-1])
			i--;
	}

	t = (x[i] + (delta*(y[i-1]-y[i+1]))/(2*(y[i+1]-2*y[i]+y[i-1]))) + 0.5; //0.5=approximation
	t_value = gpu_edge_comparison(edge, ground_truth, t, width, height);
	
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
		return gpu_find_threshold_optimized(iteration+1, t, delta*K+0.5, tolerance, K,
			edge, ground_truth, width, height, comparison, threshold, similarity);
}

__global__ void gpu_binarization(int* gpu_edge, int* gpu_binarized, int width, int height, int threshold)
{
	for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<width*height; i += blockDim.x * gridDim.x)
		gpu_binarized[i] = (gpu_edge[i] >= threshold) ? 1 : 0;
}

__inline__ __device__
int3 warp_reduce_sum_triple(int3 vals)
{
	for(int offset = warpSize/2; offset > 0; offset /= 2)
	{
		vals.x += __shfl_down(vals.x, offset);
		vals.y += __shfl_down(vals.y, offset);
		vals.z += __shfl_down(vals.z, offset);
	}	

	return vals;
}

__global__ void device_edge_comparison(int *edge, int* gt, int* success, int* false_pos, int* false_neg, int threshold, int n)
{
	int3 sums = make_int3(0,0,0);
	
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		if(edge[i] >= threshold && gt[i] == 1) //success
			sums.x++;
		else if(edge[i] >= threshold && gt[i] == 0) //false positive
			sums.y++;
		else if(edge[i] < threshold && gt[i] == 1) //false negative
			sums.z++;
	}

	sums = warp_reduce_sum_triple(sums);

	if (threadIdx.x % warpSize == 0)
	{
		atomicAdd(success, sums.x);
		atomicAdd(false_pos, sums.y);
		atomicAdd(false_neg, sums.z);
	}
}
float gpu_edge_comparison(int* dev_edge, int* dev_ground_truth, int threshold, int width, int height)
{
	int success, false_pos, false_neg;
	int* dev_success, * dev_false_pos, * dev_false_neg;
	int size = sizeof(int);

	cudaMalloc(&dev_success, size);
	cudaMalloc(&dev_false_pos, size);
	cudaMalloc(&dev_false_neg, size);

	cudaMemset(dev_success, 0, size);
	cudaMemset(dev_false_pos, 0, size);
	cudaMemset(dev_false_neg, 0, size);

	device_edge_comparison<<<BLOCKS, THREADS>>>(dev_edge, dev_ground_truth, dev_success, dev_false_pos, dev_false_neg, threshold, width*height);

	cudaMemcpy(&success, dev_success, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&false_pos, dev_false_pos, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&false_neg, dev_false_neg, size, cudaMemcpyDeviceToHost);


	cudaFree(dev_success);
	cudaFree(dev_false_pos);
	cudaFree(dev_false_neg);

	if(success == 0) //avoid 0/0
		return 0.0;
	return (float)success/(success + false_neg + false_pos);
}

__global__ void device_histogram(int* matrix, int* hist, int n)
{
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		atomicAdd(&hist[matrix[i]], 1);
}

int* gpu_histogram(int* matrix, int width, int height)
{
	int* hist = NULL;
	int* dev_hist = NULL;
	int size = 256*sizeof(int);

	hist = (int*)malloc(size);
	cudaMalloc(&dev_hist, size);
	cudaMemset(dev_hist, 0, size);

	device_histogram<<<BLOCKS, THREADS>>>(matrix, dev_hist, width*height);

	cudaMemcpy(hist, dev_hist, size, cudaMemcpyDeviceToHost);


	cudaFree(dev_hist);

	return hist;
}