#include "edge_detectors_gpu.cuh"


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