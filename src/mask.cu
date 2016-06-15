/***************************************************************************
 *
 * Author : "Huayou SU, Wen WEN, Xiaoli DU, Dongsheng LI"
 * Parallel and Distributed Processing Laboratory of NUDT
 * Author : "Maofu LIAO"
 * Department of Cell Biology, Harvard Medical School
 *
 * This file is the GPU implemenation of softmask in GeRelio.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/


#include "src/mask.h"
#include "src/math_function.h"

//extern __shared__ double smem_d[];

static __global__ void softMaskOutsideMap_kernel(double*, double, double, double*, double , int , int, int , int , int, int);

static __global__ void softMaskOutsideMap_kernel(double* vol, double radius, double cosine_width, double* Mnoise, double radius_p, int xdim, int ydim, int zdim, int xinit, int yinit, int zinit)
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int offset;
	int tid = tid_y * blockDim.x + tid_x;
	int image_size = zdim * ydim * xdim;
	int block_size = blockDim.x * blockDim.y;
	int nr_loops = (image_size + block_size - 1) / block_size;
	long int kp, ip, jp;
	offset = blockIdx.x * image_size;
	double r, raisedcos;
	__shared__ double sum_bg[BLOCK_SIZE];
	__shared__ double sum[BLOCK_SIZE];
	sum_bg[tid] = 0;
	sum[tid] = 0;

	if (Mnoise == NULL)
	{
		for (int i = 0; i < nr_loops; i++)
		{
			if (tid < image_size)
			{
				jp = ((tid % xdim) + xinit);
				ip = ((tid / xdim) + yinit);
				kp = 0;
				r = sqrt((double)(kp * kp + ip * ip + jp * jp));

				if (r < radius)
					;
				else if (r > radius_p)
				{
					sum[tid_y * blockDim.x + tid_x]    += 1.;
					sum_bg[tid_y * blockDim.x + tid_x] += vol[offset + (kp - zinit) * xdim * ydim + (ip - yinit) * xdim + (jp - xinit)];
				}
				else
				{
					raisedcos = 0.5 + 0.5 * cos(PI * (radius_p - r) / cosine_width);
					sum[tid_y * blockDim.x + tid_x] += raisedcos;
					sum_bg[tid_y * blockDim.x + tid_x] += raisedcos * vol[offset + (kp - zinit) * xdim * ydim + (ip - yinit) * xdim + (jp - xinit)];
				}
			}
			tid += block_size;
		}
	}
	////////////TODO: refuction the sum of sum and sum_bg
	// do reduction in shared mem
	__syncthreads();
	tid = tid_y * blockDim.x + tid_x;
	for (unsigned int s = (blockDim.x / 2); s > 0; s = (s >> 1))
	{
		if (tid < s)
		{
			sum[tid] += sum[tid + s];
			sum_bg[tid] += sum_bg[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0)
	{
		sum_bg[0] /= sum[0];
	}
	__syncthreads();

	// Calculate average background value
	tid = tid_y * blockDim.x + tid_x;
	for (int i = 0; i < nr_loops; i++)
	{
		if (tid < image_size)
		{
			jp = (tid % xdim + xinit);
			ip = (tid / xdim) + yinit;
			kp = (tid / (xdim * ydim) + zinit);
			r = sqrt((double)(kp * kp + ip * ip + jp * jp));
			if (r > radius_p && r >= radius)
			{
				vol[offset + (kp - zinit)*xdim * ydim + (ip - yinit)*xdim + (jp - xinit)] = (Mnoise == NULL) ? sum_bg[0] : Mnoise[offset + (kp - zinit) * xdim * ydim + (ip - yinit) * xdim + (jp - xinit)];
			}
			else if (r <= radius_p && r >= radius)
			{
				raisedcos = 0.5 + 0.5 * cos(PI * (radius_p - r) / cosine_width);
				double add = (Mnoise == NULL) ?  sum_bg[0] : Mnoise[offset + (kp - zinit) * xdim * ydim + (ip - yinit) * xdim + (jp - xinit)];
				vol[offset + (kp - zinit)*xdim * ydim + (ip - yinit)*xdim + (jp - xinit)] = (1 - raisedcos) * vol[offset + (kp - zinit) * xdim * ydim + (ip - yinit) * xdim + (jp - xinit)] + raisedcos * add;
			}
		}
		tid += block_size;
	}

}
template <typename T>
void softMaskOutsideMap_gpu(T* vol, double radius, double cosine_width, T* Mnoise, int nr_images, int xdim, int ydim, int zdim)
{
	int zinit = FIRST_XMIPP_INDEX(zdim);
	int yinit = FIRST_XMIPP_INDEX(ydim);
	int xinit = FIRST_XMIPP_INDEX(xdim);
	//vol.setXmippOrigin();
	double radius_p;
	if (radius < 0)
	{
		radius = (double)xdim / 2.;
	}
	radius_p = radius + cosine_width;

	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim(nr_images, 1, 1);
	softMaskOutsideMap_kernel<<< gridDim, blockDim>>>(vol,  radius,  cosine_width, Mnoise, radius_p, xdim, ydim, zdim, xinit, yinit, zinit);
}

template void softMaskOutsideMap_gpu<double>(double* vol, double radius, double cosine_width, double* Mnoise, int nr_images, int xdim, int ydim, int zdim);





