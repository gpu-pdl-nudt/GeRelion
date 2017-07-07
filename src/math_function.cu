/***************************************************************************
 *
 * Author : "Huayou SU, Wen WEN, Xiaoli DU, Dongsheng LI"
 * Parallel and Distributed Processing Laboratory of NUDT
 * Author : "Maofu LIAO"
 * Department of Cell Biology, Harvard Medical School
 *
 * This file is the major GPU program of GeRelio, we put most of the functions and 
 * GPU kernels in this file.
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
#include "src/math_function.h"
#include "src/args.h"

#include <stdio.h>
#include <fstream>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <complex>
#include <fstream>
#include <typeinfo>


#define NUM_CTF_PARAMETERS 9*8

__constant__ DOUBLE ctf_related_parameters_D[NUM_CTF_PARAMETERS];

//#ifndef FLOAT_PRECISION
#if !FLOAT_PRECISION && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	}
	while (assumed != old);

	return __longlong_as_double(old);
}
#endif
__device__ DOUBLE realWRAP_kernel(DOUBLE x,  DOUBLE x0, DOUBLE xF)
{
	return (((x) >= (x0) && (x) <= (xF)) ? (x) : ((x) < (x0)) \
	        ? ((x) - (int)(((x) - (x0)) / ((xF) - (x0)) - 1) * ((xF) - (x0))) : \
	        ((x) - (int)(((x) - (xF)) / ((xF) - (x0)) + 1) * ((xF) - (x0))));
}

extern __shared__ DOUBLE  xp_yp_array[ ];

//template <typename T>
__global__ void applyGeometry_2D_kernel(const  DOUBLE* __restrict__  V1, DOUBLE* V2, DOUBLE* Aref_matrix, DOUBLE* exp_offset_D,  bool wrap, int xdim, int ydim, int image_size, DOUBLE outside = 0)
{
	int tid = threadIdx.x;
	int i = blockIdx.x;
	int im_id = blockIdx.y;
	int pixel_id = tid;

	DOUBLE* xp_dim = (DOUBLE*) xp_yp_array;
	DOUBLE* yp_dim = (DOUBLE*) &xp_dim[xdim];
	DOUBLE* Aref = (DOUBLE*) &yp_dim[xdim];

	if (tid < 9)
	{
		Aref[tid] = Aref_matrix[tid + im_id * 9];
	}
	//exp_offset for x and y is zero means the matrix A is indentity
	if (pixel_id >= xdim || (exp_offset_D[im_id * 2] == 0. && exp_offset_D[im_id * 2 + 1] == 0.))
	{
		return;
	}

	// For scalings the output matrix is resized outside to the final
	// size instead of being resized inside the routine with the
	// same size as the input matrix

	int m1, n1, m2, n2;
	int x, y;
	DOUBLE xp, yp;
	DOUBLE minxp, minyp, maxxp, maxyp;
	int cen_x, cen_y, cen_xp, cen_yp;
	DOUBLE wx, wy;

	// Find center and limits of image
	cen_y  = (int)(ydim / 2);
	cen_x  = (int)(xdim / 2);
	cen_yp = (int)(ydim / 2);
	cen_xp = (int)(xdim / 2);
	minxp  = -cen_xp;
	minyp  = -cen_yp;
	maxxp  = xdim - cen_xp - 1;
	maxyp  = ydim - cen_yp - 1;

	x = -cen_x;
	y = i - cen_y;
	xp = (DOUBLE)x * Aref[0] + (DOUBLE)y * Aref[1] + Aref[2];
	yp = (DOUBLE)x * Aref[3] + (DOUBLE)y * Aref[4] + Aref[5];
	// Calculate this position in the input image according to the
	// geometrical transformation
	// they are related by
	// coords_output(=x,y) = A * coords_input (=xp,yp)
	if (tid == 0)
	{
		for (int j = 0 ; j < xdim; j ++)
		{
			if (wrap)
			{
				if (xp < minxp - XMIPP_EQUAL_ACCURACY ||
				        xp > maxxp + XMIPP_EQUAL_ACCURACY)
				{
					xp = realWRAP_kernel(xp, minxp - 0.5, maxxp + 0.5);
				}

				if (yp < minyp - XMIPP_EQUAL_ACCURACY ||
				        yp > maxyp + XMIPP_EQUAL_ACCURACY)

				{
					yp = realWRAP_kernel(yp, minyp - 0.5, maxyp + 0.5);
				}
			}
			xp_dim[j] = xp;
			yp_dim[j] = yp;
			xp += Aref[0];
			yp += Aref[3];
		}
	}
	__syncthreads();

	for (int index = tid ; index < xdim;  index += blockDim.x)
	{
		bool interp;
		DOUBLE tmp;
		interp = true;
		xp = xp_dim[index] ;
		yp = yp_dim[index] ;
		if (xp < minxp - XMIPP_EQUAL_ACCURACY ||
		        xp > maxxp + XMIPP_EQUAL_ACCURACY)
		{
			interp = false;
		}

		if (yp < minyp - XMIPP_EQUAL_ACCURACY ||
		        yp > maxyp + XMIPP_EQUAL_ACCURACY)
		{
			interp = false;
		}
		if (interp)
		{
			// Linear interpolation

			// Calculate the integer position in input image, be careful
			// that it is not the nearest but the one at the top left corner
			// of the interpolation square. Ie, (0.7,0.7) would give (0,0)
			// Calculate also weights for point m1+1,n1+1
			wx = xp + cen_xp;
			m1 = (int) wx;
			wx = wx - m1;
			m2 = m1 + 1;
			wy = yp + cen_yp;
			n1 = (int) wy;
			wy = wy - n1;
			n2 = n1 + 1;

			// m2 and n2 can be out by 1 so wrap must be check here
			if (wrap)
			{
				if (m2 >= xdim)
				{
					m2 = 0;
				}
				if (n2 >= ydim)
				{
					n2 = 0;
				}
			}

			// Perform interpolation
			// if wx == 0 means that the rightest point is useless for this
			// interpolation, and even it might not be defined if m1=xdim-1
			// The same can be said for wy.
			tmp  = (DOUBLE)((1 - wy) * (1 - wx) * V1[im_id * image_size + n1 * xdim + m1]);

			if (wx != 0 && m2 < xdim)
			{
				tmp += (DOUBLE)((1 - wy) * wx * V1[im_id * image_size + n1 * xdim + m2]);
			}

			if (wy != 0 && n2 < ydim)
			{
				tmp += (DOUBLE)(wy * (1 - wx) * V1[im_id * image_size + n2 * xdim + m1]);

				if (wx != 0 && m2 < xdim)
				{
					tmp += (DOUBLE)(wy * wx * V1[im_id * image_size + n2 * xdim + m2]);
				}
			}
			V2[im_id * image_size + i * xdim + index] = tmp;
		}
		else
		{
			V2[im_id * image_size + i * xdim + index] = outside;
		}

	}
}
//template <typename T>
void selfTranslate_gpu(DOUBLE* V1,
                       DOUBLE*  Aref_matrxi,
                       DOUBLE* exp_old_offset_D,
                       int dim, int  xdim, int ydim, int image_size,  int nr_image,
                       bool wrap, DOUBLE outside)
{
	int block_size;
	if (xdim < 32)
	{
		block_size = 32;
	}
	else if (xdim < 64)
	{
		block_size = 64;
	}
	else
	{
		block_size = 128;
	}
	dim3 dimBlock(block_size, 1, 1);
	dim3 dimGrid(ydim , nr_image, 1);
	int shared_mem_size = xdim * sizeof(DOUBLE) * 2 + 9 * sizeof(DOUBLE);
	DOUBLE* temp;
	cudaMalloc((void**)&temp, image_size * nr_image * sizeof(DOUBLE));
	cudaMemcpy(temp, V1, image_size * nr_image * sizeof(DOUBLE), cudaMemcpyDeviceToDevice);
	applyGeometry_2D_kernel <<< dimGrid , dimBlock, shared_mem_size>>>(temp,  V1, Aref_matrxi, exp_old_offset_D, wrap,  xdim,  ydim,  image_size,  outside);
	cudaFree(temp);
}
/*template void selfTranslate_gpu<DOUBLE>(DOUBLE* V1,
                                        DOUBLE*  Aref_matrxi,
                                        DOUBLE* exp_old_offset_D,
                                        int dim, int  xdim, int ydim, int image_size,  int nr_image,
                                        bool wrap = WRAP, DOUBLE outside = 0);
template void selfTranslate_gpu<float>(float* V1,
                                       DOUBLE*  Aref_matrxi,
                                       DOUBLE* exp_old_offset_D,
                                       int dim, int xdim, int ydim, int image_size,  int nr_image,
                                       bool wrap = WRAP, float outside = 0);
*/
//template <typename T>
__global__ void do_norm_correction_kernel(DOUBLE* image_D, DOUBLE* normcorr_D, int image_size, DOUBLE avg_norm_correction)
{
	int tid = threadIdx.x;
	int im_id = blockIdx.y;
	int pixel_id = tid + blockIdx.x * blockDim.x;

	if (pixel_id >= image_size)
	{
		return;
	}

	image_D[pixel_id + im_id * image_size] =   image_D[pixel_id + im_id * image_size] * (avg_norm_correction / normcorr_D[im_id ]);

}
//template <typename T>
void do_norm_correction_gpu(DOUBLE* image_D, DOUBLE* normcorr_D, int image_size, int exp_nr_images, DOUBLE avg_norm_correction)
{
	int blk_x = (image_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128;
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(blk_x, exp_nr_images, 1);
	do_norm_correction_kernel <<< dimGrid, dimBlock>>>(image_D, normcorr_D, image_size , avg_norm_correction);

}
//template void do_norm_correction_gpu<DOUBLE>(DOUBLE* image_D, DOUBLE* normcorr_D, int image_size, int exp_nr_images, DOUBLE avg_norm_correction);
//template void do_norm_correction_gpu<float>(float* image_D, DOUBLE* normcorr_D, int image_size, int exp_nr_images, DOUBLE avg_norm_correction);
//template <typename T>
__global__ void scal_images(DOUBLE* alpha, DOUBLE* inout, int image_size)
{

	int tid = threadIdx.x;
	int offset = blockIdx.x * image_size;
	int nr_loops = image_size / blockDim.x + 1;
	for (int i = 0; i < nr_loops; i++)
	{

		if ((tid + i * blockDim.x) < image_size)
		{
			inout[tid + i * blockDim.x + offset] =  inout[tid + i * blockDim.x + offset] * alpha[blockIdx.x ];
		}
	}
}


//template <typename T>
void relion_gpu_scal(const int N,  DOUBLE* alpha, DOUBLE* X, int stride)
{
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(N, 1, 1);
	scal_images <<< dimGrid, dimBlock>>>(alpha, X, stride);
}


//template <typename T>
__global__ void shift_1D_kernel(DOUBLE* in, DOUBLE* out, int xdim, int shift)
{

	int tid = threadIdx.x;
	int pos;

	if (tid < xdim)
	{
		pos = tid + shift;
		pos = (pos < 0) ? (pos + xdim) : (pos >= xdim ? (pos - xdim) : pos);
		out[pos + blockIdx.x * xdim] =  in[tid + blockIdx.x * xdim];
	}
}

//template <typename T>
__global__ void shift_2D_kernel(DOUBLE* in, DOUBLE* out, int xdim, int ydim, int xshift, int yshift)
{

	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;

	int posx, posy;
	int nr_loopx, nr_loopy;
	nr_loopx = (xdim + blockDim.x - 1) / blockDim.x;
	nr_loopy = (ydim + blockDim.y - 1) / blockDim.y;

	//The kernel codes for 2D  threak block configuration
	for (int j = 0; j < nr_loopy; j++)
	{
		posy = (tid_y + j * blockDim.y + yshift >= ydim) ? (tid_y + j * blockDim.y + yshift - ydim) : (tid_y + j * blockDim.y + yshift);
		for (int i = 0 ; i < nr_loopx; i++)
		{
			posx = ((tid_x + i * blockDim.x + xshift) >= xdim) ? (tid_x + i * blockDim.x + xshift - xdim) : (tid_x + i * blockDim.x + xshift);
			if ((tid_y + j * blockDim.y) < ydim && (tid_x + i * blockDim.x) < xdim)
			{
				out[posx + posy * xdim + blockIdx.x * xdim * ydim] = in[(tid_x + i * blockDim.x) + (tid_y + j * blockDim.y) * xdim + blockIdx.x * xdim * ydim];
			}
		}

	}

}

//template <typename T>
__global__ void shift_3D_kernel(DOUBLE* in, DOUBLE* out, int xdim, int ydim, int zdim, int xshift, int yshift, int zshift)
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;

	int nr_loopx, nr_loopy;
	nr_loopx = (xdim + blockDim.x - 1) / blockDim.x;
	nr_loopy = (ydim + blockDim.y - 1) / blockDim.y;
	int posx, posy, posz;
	for (int k = 0; k < zdim; k++)
	{
		posz = (k + zshift >= zdim) ? (k + zshift - zdim) : (k + zshift);

		for (int j = 0; j < nr_loopy; j++)
		{
			posy = ((tid_y + j * blockDim.y + yshift) >= ydim) ? (tid_y + j * blockDim.y + yshift - ydim) : (tid_y + j * blockDim.y + yshift);
			for (int i = 0 ; i < nr_loopx; i++)
			{
				posx = ((tid_x + i * blockDim.x + xshift) >= xdim) ? (tid_x + i * blockDim.x + xshift - xdim) : (tid_x + i * blockDim.x + xshift);
				if ((tid_y + j * blockDim.y) < ydim && (tid_x + i * blockDim.x) < xdim)
				{
					out[posx + posy * xdim + posz * xdim * ydim + blockIdx.x * xdim * ydim * zdim] = in[(tid_x + i * blockDim.x) + (tid_y + j * blockDim.y) * xdim * +k * xdim * ydim + blockIdx.x * xdim * ydim * zdim];
				}
			}
		}
	}
}
//template <typename T>
void centerFFT_gpu(DOUBLE* in, DOUBLE* out, int nr_images, int dim, int xdim, int ydim, int zdim, bool forward)
{
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimBlock2D(BLOCK_X, BLOCK_Y, 1);
	dim3 dimGrid(nr_images, 1, 1);
	if (dim == 1)
	{
		int shift = (int)(xdim / 2);
		if (!forward)
		{
			shift = -shift;
		}
		shift_1D_kernel <<< dimGrid, dimBlock>>>(in, out, xdim, shift);
	}
	else if (dim == 2)
	{
		int xshift = xdim / 2;
		int yshift = ydim / 2;
		if (!forward)
		{
			xshift = -xshift;
			yshift = -yshift;
		}
		shift_2D_kernel <<< dimGrid, dimBlock2D>>>(in, out, xdim, ydim, xshift, yshift);
	}
	else if (dim == 3)
	{
		int xshift = xdim / 2;
		int yshift = ydim / 2;
		int zshift = zdim / 2;
		if (!forward)
		{
			xshift = -xshift;
			yshift = -yshift;
			zshift = -zshift;
		}
		shift_3D_kernel <<< dimGrid, dimBlock2D>>>(in, out, xdim, ydim, zdim, xshift, yshift, zshift);
	}
	else
	{
		REPORT_ERROR("CenterFFT ERROR: Dimension should be 1, 2 or 3");
	}
}

// Explicit instantiation
//template void centerFFT_gpu<DOUBLE>(DOUBLE* in, DOUBLE* out, int nr_images, int dim, int xdim, int ydim, int zdim, bool forward);
//template void centerFFT_gpu<float>(float* in, float* out, int nr_images, int dim, int xdim, int ydim, int zdim, bool forward);

//template <typename T>
__global__ void calculate_local_sqrtXi2_kernel(CUFFT_COMPLEX* local_Fimgs_D, DOUBLE* exp_local_sqrtXi2_D, int image_size)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int n_loop;
	DOUBLE squr_sum;
	n_loop = (image_size + blockDim.x - 1) / blockDim.x;
	__shared__ DOUBLE local_sum[BLOCK_SIZE];
	local_sum[tid] = 0;
	squr_sum = 0;
	for (int i = 0 ; i < n_loop; i++)
	{
		if ((tid + i * blockDim.x) < image_size)
		{
			CUFFT_COMPLEX a = local_Fimgs_D[tid + i * blockDim.x + bid * image_size];
			squr_sum += (a.x * a.x) + (a.y * a.y);
		}
	}
	local_sum[tid] = squr_sum;
	__syncthreads();

	for (unsigned int s = (blockDim.x / 2); s > 0; s = (s >> 1))
	{
		if (tid < s)
		{
			local_sum[tid] += local_sum[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0)
	{
		exp_local_sqrtXi2_D[bid] = sqrt(local_sum[0]);
	}

}
//template <typename T>
void calculate_local_sqrtXi2_gpu(CUFFT_COMPLEX* local_Fimgs_D, DOUBLE* exp_local_sqrtXi2_D, int nr_images, int image_size)
{
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim(nr_images, 1, 1);
	calculate_local_sqrtXi2_kernel <<< gridDim, blockDim>>>(local_Fimgs_D, exp_local_sqrtXi2_D, image_size);

}

//template void calculate_local_sqrtXi2_gpu<CUFFT_COMPLEX >(CUFFT_COMPLEX * local_Fimgs_D, DOUBLE* exp_local_sqrtXi2_D, int nr_images, int image_size);
__global__ void calculate_Minvsigma2_kernel(DOUBLE* exp_Minvsigma2_D, int* local_myMresol_D, DOUBLE* sigma2_noise_D, int* group_id_D, DOUBLE sigma2_fudge, int image_size, int myMresol_size, int noise_size_of_group)
{
	int tid = threadIdx.x;
	int num_loop, ires;
	int group_id = group_id_D[blockIdx.x];
	DOUBLE localMinvsigma2;
	num_loop = (image_size + blockDim.x - 1) / blockDim.x;
	for (int i = 0; i < num_loop; i++)
	{
		if (tid < myMresol_size)
		{
			ires =  local_myMresol_D[tid];
			localMinvsigma2 = 1. / (sigma2_fudge * sigma2_noise_D[group_id * noise_size_of_group + ires]);
			if (ires > 0)
			{
				exp_Minvsigma2_D[blockIdx.x * image_size + tid] = localMinvsigma2;
			}
		}
		tid += blockDim.x;
	}
}
void calculate_Minvsigma2_gpu(DOUBLE* exp_Minvsigma2_D, int* local_myMresol_D, DOUBLE* sigma2_noise_D, int* group_id_D,  DOUBLE sigma2_fudge, int nr_images, int image_size, int myMresol_size, int noise_size_of_group)
{
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim(nr_images, 1, 1);
	calculate_Minvsigma2_kernel <<< gridDim, blockDim>>>(exp_Minvsigma2_D,  local_myMresol_D, sigma2_noise_D, group_id_D,  sigma2_fudge,  image_size, myMresol_size, noise_size_of_group);
}

//The ctfref data is stored continues for each particle
__global__ void calculate_frefctf_Mctf_kernel(CUFFT_COMPLEX * frefctf_D,
                                              const CUFFT_COMPLEX * __restrict__ fref_D,
                                              const DOUBLE* __restrict__  mctf_D,
                                              DOUBLE* mctf_out_D,
                                              const DOUBLE* __restrict__ myscale_D,
                                              int nr_oversampled_rot,
                                              int image_size,
                                              int nr_images,
                                              bool do_ctf_correction_and_refs_are_ctf_corrected,
                                              bool do_scale_correction)
{

	int tid = threadIdx.x;
	int imageIdx = blockIdx.y;
	int offset_image = imageIdx * image_size ;
	int offset_rot = blockIdx.x * image_size;
	int output_offset = offset_rot + imageIdx * gridDim.x * image_size;

	DOUBLE frefctf_real, frefctf_imag;
	for (int i = tid ; i < image_size; i += BLOCK_SIZE)
	{

		frefctf_real = fref_D[i + offset_rot].x;
		frefctf_imag = fref_D[i + offset_rot].y;

		if (do_ctf_correction_and_refs_are_ctf_corrected)
		{
			frefctf_real *= mctf_D[i + offset_image];
			frefctf_imag *= mctf_D[i + offset_image];
		}
		if (do_scale_correction)
		{
			frefctf_real *= myscale_D[imageIdx];
			frefctf_imag *= myscale_D[imageIdx];
			mctf_out_D[i + output_offset] =  mctf_D[i + offset_image] * myscale_D[imageIdx];
		}

		frefctf_D[i + output_offset].x = frefctf_real;
		frefctf_D[i + output_offset].y = frefctf_imag;
	}

}

void calculate_frefctf_Mctf_gpu(CUFFT_COMPLEX * frefctf_D,
                                CUFFT_COMPLEX * fref_D,
                                DOUBLE* mctf_D,
                                DOUBLE* mctf_out_D,
                                DOUBLE* myscale_D,
                                int nr_orients,
                                int nr_oversampled_rot,
                                int nr_ipart,
                                int image_size,
                                bool do_ctf_correction_and_refs_are_ctf_corrected,
                                bool do_scale_correction)
{
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(nr_oversampled_rot * nr_orients, nr_ipart, 1);
	calculate_frefctf_Mctf_kernel <<< dimGrid, dimBlock>>>(frefctf_D,
	                                                       fref_D,
	                                                       mctf_D,
	                                                       mctf_out_D,
	                                                       myscale_D,
	                                                       nr_oversampled_rot,
	                                                       image_size,
	                                                       nr_ipart,
	                                                       do_ctf_correction_and_refs_are_ctf_corrected,
	                                                       do_scale_correction);

}


__global__ void calculate_frefctf_kernel(CUFFT_COMPLEX * frefctf_D,
                                         const  CUFFT_COMPLEX * __restrict__  fref_D,
                                         const DOUBLE* __restrict__ mctf_D,
                                         const DOUBLE* __restrict__ myscale_D,
                                         int nr_oversampled_rot,
                                         int image_size,
                                         int nr_images,
                                         bool do_ctf_correction_and_refs_are_ctf_corrected,
                                         bool do_scale_correction)
{

	int tid = threadIdx.x;
	int imageIdx = blockIdx.y;
	int offset_image = imageIdx * image_size ;
	int offset_rot = blockIdx.x * image_size;
	int output_offset = offset_rot + imageIdx * gridDim.x * image_size;

	DOUBLE frefctf_real, frefctf_imag;
	for (int i = tid; i < image_size ; i += blockDim.x)
	{
		frefctf_real = fref_D[i + offset_rot].x;
		frefctf_imag = fref_D[i + offset_rot].y;

		if (do_ctf_correction_and_refs_are_ctf_corrected)
		{
			frefctf_real *= mctf_D[i + offset_image];
			frefctf_imag *= mctf_D[i + offset_image];
		}
		if (do_scale_correction)
		{
			frefctf_real *= myscale_D[imageIdx];
			frefctf_imag *= myscale_D[imageIdx];
		}

		frefctf_D[i + output_offset].x = frefctf_real;
		frefctf_D[i + output_offset].y = frefctf_imag;
	}

}

__global__ void calculate_frefctf_do_scale_correction_kernel(CUFFT_COMPLEX * frefctf_D,
                                                             const  CUFFT_COMPLEX * __restrict__  fref_D,
                                                             const DOUBLE* __restrict__ mctf_D,
                                                             const DOUBLE* __restrict__ myscale_D,
                                                             int nr_oversampled_rot,
                                                             int image_size,
                                                             int nr_images)
{

	int tid = threadIdx.x;
	int imageIdx = blockIdx.y;
	int offset_rot = blockIdx.x * image_size;
	int output_offset = offset_rot + imageIdx * gridDim.x * image_size;

	DOUBLE frefctf_real, frefctf_imag;
	for (int i = tid; i < image_size ; i += blockDim.x)
	{
		frefctf_real = fref_D[i + offset_rot].x;
		frefctf_imag = fref_D[i + offset_rot].y;

		frefctf_real *= myscale_D[imageIdx];
		frefctf_imag *= myscale_D[imageIdx];


		frefctf_D[i + output_offset].x = frefctf_real;
		frefctf_D[i + output_offset].y = frefctf_imag;
	}

}

__global__ void calculate_frefctf_all_kernel(CUFFT_COMPLEX * frefctf_D,
                                             const  CUFFT_COMPLEX * __restrict__  fref_D,
                                             const DOUBLE* __restrict__ mctf_D,
                                             const DOUBLE* __restrict__ myscale_D,
                                             int nr_oversampled_rot,
                                             int image_size,
                                             int nr_images)
{

	int tid = threadIdx.x;
	int offset_rot = blockIdx.x * image_size;
	int output_offset;

	DOUBLE frefctf_real, frefctf_imag;
	for (int i = tid; i < image_size ; i += blockDim.x)
	{
		frefctf_real = fref_D[i + offset_rot].x;
		frefctf_imag = fref_D[i + offset_rot].y;
		for (int imageIdx = 0; imageIdx < nr_images; imageIdx++)
		{
			output_offset = offset_rot + imageIdx * gridDim.x * image_size;
			//imageIdx
			frefctf_D[i + output_offset].x = frefctf_real * mctf_D[i + imageIdx * image_size] * myscale_D[imageIdx]; //frefctf_real;
			frefctf_D[i + output_offset].y = frefctf_imag * mctf_D[i + imageIdx * image_size] * myscale_D[imageIdx];
		}
	}

}

__global__ void calculate_frefctf_do_ctf_correction_kernel(CUFFT_COMPLEX * frefctf_D,
                                                           const  CUFFT_COMPLEX * __restrict__  fref_D,
                                                           const DOUBLE* __restrict__ mctf_D,
                                                           const DOUBLE* __restrict__ myscale_D,
                                                           int nr_oversampled_rot,
                                                           int image_size,
                                                           int nr_images)
{

	int tid = threadIdx.x;
	int imageIdx = blockIdx.y;
	int offset_image = imageIdx * image_size ;
	int offset_rot = blockIdx.x * image_size;
	int output_offset = offset_rot + imageIdx * gridDim.x * image_size;

	for (int i = tid; i < image_size ; i += blockDim.x)
	{

		frefctf_D[i + output_offset].x = fref_D[i + offset_rot].x * mctf_D[i + offset_image];
		frefctf_D[i + output_offset].y = fref_D[i + offset_rot].y * mctf_D[i + offset_image];

	}

}

void  calculate_frefctf_gpu(CUFFT_COMPLEX * frefctf_D,
                            CUFFT_COMPLEX * fref_D,
                            DOUBLE* exp_local_Fctfs_D,
                            DOUBLE* myscale_D,
                            int nr_images,
                            int nr_orients,
                            int nr_oversampled_rot,
                            int image_size,
                            bool do_ctf_correction_and_refs_are_ctf_corrected,
                            bool do_scale_correction)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(nr_oversampled_rot * nr_orients, nr_images, 1);
	if (do_ctf_correction_and_refs_are_ctf_corrected && do_scale_correction)
	{
		dim3 dimGrid2(nr_oversampled_rot * nr_orients, 1, 1);
		calculate_frefctf_all_kernel <<< dimGrid2, dimBlock>>>(frefctf_D,
		                                                       fref_D,
		                                                       exp_local_Fctfs_D,
		                                                       myscale_D,
		                                                       nr_oversampled_rot,
		                                                       image_size,
		                                                       nr_images);
	}
	else if (do_ctf_correction_and_refs_are_ctf_corrected)
	{
		calculate_frefctf_do_ctf_correction_kernel <<< dimGrid, dimBlock>>>(frefctf_D,
		                                                                    fref_D,
		                                                                    exp_local_Fctfs_D,
		                                                                    myscale_D,
		                                                                    nr_oversampled_rot,
		                                                                    image_size,
		                                                                    nr_images);
	}
	else if (do_scale_correction)
	{
		calculate_frefctf_do_scale_correction_kernel <<< dimGrid, dimBlock>>>(frefctf_D,
		                                                                      fref_D,
		                                                                      exp_local_Fctfs_D,
		                                                                      myscale_D,
		                                                                      nr_oversampled_rot,
		                                                                      image_size,
		                                                                      nr_images);

	}


}
extern __shared__ DOUBLE  thr_wsum_array[ ];

__global__ void calculate_wdiff2_sumXA_Meta_total_kernel(
    const  CUFFT_COMPLEX * __restrict__ frefctf_D,
    const  CUFFT_COMPLEX * __restrict__ Fimg_shift_D,
    const DOUBLE* __restrict__ weight_D,
    const int* __restrict__ weight_index,
    const int* __restrict__ isSignificant_D,
    int Mresol_fine_size,
    bool do_scale_correction,
    int image_size,
    int nr_images,
    int nr_orients,
    int nr_oversampled_rot,
    int nr_trans,
    int nr_oversampled_trans,
    int nr_valid_orients,
    DOUBLE* thr_wsum_sigma2_noise_D,
    DOUBLE* thr_wsum_norm_correction_D,
    DOUBLE* thr_wsum_scale_correction_XA_D,
    DOUBLE* thr_wsum_scale_correction_AA_D,
    DOUBLE* data_vs_prior_class_D,
    int* mresol_fine_D,
    int* group_id_D,
    int thr_wsum_size,
    int ref_dim,
    int modelorientational_prior_mode,
    int exp_nr_psi,
    DOUBLE model_prior_offset_class_x,
    DOUBLE model_prior_offset_class_y,
    DOUBLE* exp_old_offset_D,
    DOUBLE* oversampled_translations_D,
    DOUBLE* pointer_dir_nonzeroprior_D,
    DOUBLE* thr_sumw_group_D,
    DOUBLE* thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D,
    DOUBLE* exp_prior_D,
    DOUBLE* thr_wsum_pdf_direction_D
)
{

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	DOUBLE frefctf_real, frefctf_imag;
	DOUBLE shift_real, shift_imag;
	DOUBLE wdiff2 = 0.;
	DOUBLE diff_real, diff_imag;
	DOUBLE sumXA, sumA2;
	DOUBLE weight;
	int part_id, ishift_id, ires, group_id, iover_rot, orient_id, oriend_reorder_id, exp_trans_id, iover_trans_id;
	int weight_id = weight_index[bid];

	part_id = weight_id / (nr_orients * nr_oversampled_rot * nr_trans * nr_oversampled_trans);
	iover_rot = (weight_id % (nr_oversampled_rot * nr_oversampled_trans)) / (nr_oversampled_trans);
	orient_id = (weight_id % (nr_orients * nr_oversampled_rot * nr_trans * nr_oversampled_trans)) / (nr_oversampled_rot * nr_trans * nr_oversampled_trans);
	exp_trans_id = (weight_id % (nr_oversampled_rot * nr_trans * nr_oversampled_trans)) / (nr_oversampled_rot * nr_oversampled_trans);
	iover_trans_id = (weight_id % nr_oversampled_trans);
	ishift_id = part_id * nr_oversampled_trans * nr_trans +  exp_trans_id * nr_oversampled_trans + iover_trans_id;

	weight = weight_D[bid];
	group_id = group_id_D[part_id];
	oriend_reorder_id = isSignificant_D[orient_id];
	DOUBLE data_vs_prior_class;

	DOUBLE* thr_wsum_norm_correction_SHM = (DOUBLE*) thr_wsum_array;
	DOUBLE* thr_wsum_sigma2_noise_SHM = (DOUBLE*) &thr_wsum_norm_correction_SHM[BLOCK_SIZE_128];
	DOUBLE* thr_wsum_scale_correction_XA_SHM = (DOUBLE*) &thr_wsum_sigma2_noise_SHM[((thr_wsum_size + 32 - 1) / 32) * 32];
	DOUBLE* thr_wsum_scale_correction_AA_SHM = (DOUBLE*) &thr_wsum_scale_correction_XA_SHM[((thr_wsum_size + 32 - 1) / 32) * 32];

	thr_wsum_norm_correction_SHM[tid] = 0.;
	int ref_image_offset = part_id * nr_valid_orients * nr_oversampled_rot * image_size + ((oriend_reorder_id * nr_oversampled_rot + iover_rot) * image_size);
	int shift_image_offset = ishift_id * image_size;
	if (tid < thr_wsum_size)
	{
		thr_wsum_sigma2_noise_SHM[tid] = 0.;
		thr_wsum_scale_correction_XA_SHM[tid] = 0.;
		thr_wsum_scale_correction_AA_SHM[tid] = 0.;
	}
	__syncthreads();


	for (int i = tid; i < Mresol_fine_size ; i += blockDim.x)
	{

		ires = mresol_fine_D[i];
		data_vs_prior_class = data_vs_prior_class_D[ires];
		if (ires > -1)
		{
			frefctf_real = frefctf_D[i + ref_image_offset].x;
			frefctf_imag = frefctf_D[i + ref_image_offset].y;
			shift_real = Fimg_shift_D[i + shift_image_offset].x;
			shift_imag = Fimg_shift_D[i + shift_image_offset].y;

			diff_real = frefctf_real - shift_real;
			diff_imag = frefctf_imag - shift_imag;
			wdiff2 = weight * (diff_real * diff_real + diff_imag * diff_imag);

			thr_wsum_norm_correction_SHM[tid] += wdiff2;//wdiff2_D[bid * image_size + tid +i*blockDim.x];
			atomicAdd(&(thr_wsum_sigma2_noise_SHM[ires]), (DOUBLE)wdiff2);


			if (do_scale_correction)
			{
				if (data_vs_prior_class > 3.)
				{
					sumXA = frefctf_real * shift_real;
					sumXA += frefctf_imag * shift_imag;
					sumXA *= weight;
					atomicAdd(&(thr_wsum_scale_correction_XA_SHM[ires]), (DOUBLE)sumXA);

					sumA2 = frefctf_real * frefctf_real;
					sumA2 += frefctf_imag * frefctf_imag;
					sumA2 *= weight;
					atomicAdd(&(thr_wsum_scale_correction_AA_SHM[ires]), (DOUBLE)sumA2);
				}
			}
		}
	}
	__syncthreads();

	if (tid < thr_wsum_size)
	{
		atomicAdd(&(thr_wsum_sigma2_noise_D[group_id * thr_wsum_size + tid]), (DOUBLE)thr_wsum_sigma2_noise_SHM[tid]);
		atomicAdd(&(thr_wsum_scale_correction_XA_D[part_id * thr_wsum_size + tid]), (DOUBLE)thr_wsum_scale_correction_XA_SHM[tid]);
		atomicAdd(&(thr_wsum_scale_correction_AA_D[part_id * thr_wsum_size + tid]), (DOUBLE)thr_wsum_scale_correction_AA_SHM[tid]);
	}

	for (int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			thr_wsum_norm_correction_SHM[tid] += thr_wsum_norm_correction_SHM[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();

	if (tid == 0)
	{
		atomicAdd(&(thr_wsum_norm_correction_D[part_id]), (DOUBLE)thr_wsum_norm_correction_SHM[tid]);
	}



	if (tid == 32)
	{
		// Store sum of weights for this group
		atomicAdd(&(thr_sumw_group_D[group_id]), (DOUBLE)weight);
		// Store weights for this class and orientation
		atomicAdd(&(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D[0]), (DOUBLE)weight);
	}
	if (tid == 64)
	{
		if (ref_dim == 2)
		{
			// Also store weighted offset differences for prior_offsets of each class
			DOUBLE x_part = weight * (exp_old_offset_D[part_id] + oversampled_translations_D[iover_trans_id + exp_trans_id * nr_oversampled_trans]);

			atomicAdd(&(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D[2]), (DOUBLE)x_part);

			x_part = weight * (exp_old_offset_D[part_id + nr_images] + oversampled_translations_D[iover_trans_id + exp_trans_id * nr_oversampled_trans + nr_trans * nr_oversampled_trans]);
			atomicAdd(&(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D[3]), (DOUBLE)x_part);
			DOUBLE sum2 = 0;
			x_part = model_prior_offset_class_x - exp_old_offset_D[part_id] - oversampled_translations_D[iover_trans_id + exp_trans_id * nr_oversampled_trans];
			sum2  += x_part * x_part;
			x_part = model_prior_offset_class_y - exp_old_offset_D[part_id + nr_images] - oversampled_translations_D[iover_trans_id + exp_trans_id * nr_oversampled_trans + nr_trans * nr_oversampled_trans];
			sum2  += x_part * x_part;
			// Store weighted sum2 of origin offsets (in Angstroms instead of pixels!!!)
			atomicAdd(&(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D[1]), (DOUBLE) weight * sum2);
		}
		else
		{
			DOUBLE sum2 = 0;
			DOUBLE x_part =   exp_prior_D[part_id] - exp_old_offset_D[part_id] - oversampled_translations_D[iover_trans_id + exp_trans_id * nr_oversampled_trans];
			sum2  += x_part * x_part;
			x_part = exp_prior_D[part_id] - exp_old_offset_D[part_id + nr_images] - oversampled_translations_D[iover_trans_id + exp_trans_id * nr_oversampled_trans + nr_trans * nr_oversampled_trans];
			sum2  += x_part * x_part;
			// Store weighted sum2 of origin offsets (in Angstroms instead of pixels!!!)
			atomicAdd(&(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D[1]), (DOUBLE)(weight * sum2));
		}
	}
	if (tid == 96)
	{
		// Store weight for this direction of this class
		long int idir  = (orient_id / exp_nr_psi);
		if (modelorientational_prior_mode == 0) //NOPRIOR ==0
		{
			atomicAdd(&(thr_wsum_pdf_direction_D[idir]), (DOUBLE) weight);
		}
		else
		{
			// In the case of orientational priors, get the original number of the direction back
			long int mydir = pointer_dir_nonzeroprior_D[idir]; //sampling.getDirectionNumberAlsoZeroPrior(idir);
			atomicAdd(&(thr_wsum_pdf_direction_D[mydir]), (DOUBLE) weight);
		}
	}

}

void calculate_wdiff2_sumXA_Meta_total_gpu(
    CUFFT_COMPLEX * Frecctf_D,
    CUFFT_COMPLEX * Fimg_shift_D,
    DOUBLE* weight_D,
    int* weight_index,
    int*    isSignificant_D,
    int Mresol_fine_size,
    bool do_scale_correction,
    int valid_blocks,
    int image_size,
    int nr_images,
    int nr_orients,
    int nr_oversampled_rot,
    int nr_trans,
    int nr_oversampled_trans,
    int nr_valid_orients,
    DOUBLE* thr_wsum_sigma2_noise_D,
    DOUBLE* thr_wsum_norm_correction_D,
    DOUBLE* thr_wsum_scale_correction_XA_D,
    DOUBLE* thr_wsum_scale_correction_AA_D,
    DOUBLE* data_vs_prior_class_D,
    int* mresol_fine_D,
    int* group_id_D,
    int thr_wsum_size,
    int ref_dim,
    int modelorientational_prior_mode,
    int exp_nr_psi,
    DOUBLE model_prior_offset_class_x,
    DOUBLE model_prior_offset_class_y,
    DOUBLE* exp_old_offset_D,
    DOUBLE* oversampled_translations_D,
    DOUBLE* pointer_dir_nonzeroprior_D,
    DOUBLE* thr_sumw_group_D,
    DOUBLE* thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D,
    DOUBLE* exp_prior_D,
    DOUBLE* thr_wsum_pdf_direction_D
)

{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(valid_blocks, 1, 1);
	int shared_size;
	shared_size = sizeof(DOUBLE) * (BLOCK_SIZE_128 + ((thr_wsum_size + 32 - 1) / 32) * 32 * 3);


	calculate_wdiff2_sumXA_Meta_total_kernel <<< dimGrid, dimBlock, shared_size>>>(
	    Frecctf_D,
	    Fimg_shift_D,
	    weight_D,
	    weight_index,
	    isSignificant_D,
	    Mresol_fine_size,
	    do_scale_correction,
	    image_size,
	    nr_images,
	    nr_orients,
	    nr_oversampled_rot,
	    nr_trans,
	    nr_oversampled_trans,
	    nr_valid_orients,
	    thr_wsum_sigma2_noise_D,
	    thr_wsum_norm_correction_D,
	    thr_wsum_scale_correction_XA_D,
	    thr_wsum_scale_correction_AA_D,
	    data_vs_prior_class_D,
	    mresol_fine_D,
	    group_id_D,
	    thr_wsum_size,
	    ref_dim,
	    modelorientational_prior_mode,
	    exp_nr_psi,
	    model_prior_offset_class_x,
	    model_prior_offset_class_y,
	    exp_old_offset_D,
	    oversampled_translations_D,
	    pointer_dir_nonzeroprior_D,
	    thr_sumw_group_D,
	    thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D,
	    exp_prior_D,
	    thr_wsum_pdf_direction_D
	);

}

__global__ void calculate_sum_shift_img_kernel(CUFFT_COMPLEX * Fimg_D,
                                               DOUBLE* fweight_D,
                                               const  CUFFT_COMPLEX * __restrict__ Fimg_shift_nomask_D,
                                               const DOUBLE* __restrict__ Minvsigma2_D,
                                               const DOUBLE* __restrict__ weight_D,
                                               const DOUBLE* __restrict__ mctf_D,
                                               const int* __restrict__ weight_index,
                                               const int* __restrict__ isSignificant_D,
                                               int image_size,
                                               int nr_images,
                                               int nr_orients,
                                               int nr_oversampled_rot,
                                               int nr_trans,
                                               int nr_oversampled_trans,
                                               int nr_valid_orients
                                              )

{

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int part_id, ishift_id, iover_rot, orient_id, exp_trans_id, iover_trans_id;
	DOUBLE myctf, weightxinvsigma2, weight;

	DOUBLE real, imag, fweight;
	DOUBLE Minvsigma2;
	int weight_id = weight_index[bid];
	part_id = weight_id / (nr_orients * nr_oversampled_rot * nr_trans * nr_oversampled_trans);
	orient_id = (weight_id % (nr_orients * nr_oversampled_rot * nr_trans * nr_oversampled_trans)) / (nr_oversampled_rot * nr_trans * nr_oversampled_trans);
	iover_rot = (weight_id % (nr_oversampled_rot * nr_oversampled_trans)) / (nr_oversampled_trans);
	exp_trans_id = (weight_id % (nr_oversampled_rot * nr_trans * nr_oversampled_trans)) / (nr_oversampled_rot * nr_oversampled_trans);
	iover_trans_id = (weight_id % nr_oversampled_trans);
	ishift_id = part_id * nr_oversampled_trans * nr_trans +  exp_trans_id * nr_oversampled_trans + iover_trans_id;

	weight = weight_D[bid];
	int reorder_orient_id = isSignificant_D[orient_id];
	int ctf_image_offset = part_id * nr_valid_orients * nr_oversampled_rot * image_size + (isSignificant_D[orient_id] * nr_oversampled_rot + iover_rot) * image_size;
	int shfit_image_offset = ishift_id * image_size;
	for (int i = tid; i < image_size; i += blockDim.x)
	{
		real = imag = fweight = 0.;

		myctf = mctf_D[i + ctf_image_offset];
		Minvsigma2 =  Minvsigma2_D[i + part_id * image_size];
		weightxinvsigma2 = weight * myctf * Minvsigma2;

		real = Fimg_shift_nomask_D[i + shfit_image_offset].x * weightxinvsigma2;
		imag = Fimg_shift_nomask_D[i + shfit_image_offset].y * weightxinvsigma2;
		fweight = weightxinvsigma2 * myctf;

		atomicAdd(&(Fimg_D[i + (reorder_orient_id * nr_oversampled_rot + iover_rot)*image_size].x), (DOUBLE)real);
		atomicAdd(&(Fimg_D[i + (reorder_orient_id * nr_oversampled_rot + iover_rot)*image_size].y), (DOUBLE)imag);
		atomicAdd(&(fweight_D[i + (reorder_orient_id * nr_oversampled_rot + iover_rot)*image_size]), (DOUBLE)fweight);
	}

}

void calculate_sum_shift_img_gpu(CUFFT_COMPLEX * Fimg_D,
                                 DOUBLE* fweight_D,
                                 CUFFT_COMPLEX * Fimg_shift_nomask_D,
                                 DOUBLE* Minvsigma2_D,
                                 DOUBLE* weight_D,
                                 DOUBLE* mctf_D,
                                 int* weight_index,
                                 int*    isSignificant_D,
                                 int image_size,
                                 int nr_weight,
                                 int nr_images,
                                 int nr_orients,
                                 int nr_oversampled_rot,
                                 int nr_trans,
                                 int nr_oversampled_trans,
                                 int nr_valid_orients
                                )
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(nr_weight, 1, 1);
	calculate_sum_shift_img_kernel <<< dimGrid, dimBlock>>>(Fimg_D,
	                                                        fweight_D,
	                                                        Fimg_shift_nomask_D,
	                                                        Minvsigma2_D,
	                                                        weight_D,
	                                                        mctf_D,
	                                                        weight_index,
	                                                        isSignificant_D,
	                                                        image_size,
	                                                        nr_images,
	                                                        nr_orients,
	                                                        nr_oversampled_rot,
	                                                        nr_trans,
	                                                        nr_oversampled_trans,
	                                                        nr_valid_orients
	                                                       );

}

extern __shared__ DOUBLE  local_weight_array[ ];
__global__ void calculate_sum_shift_img_kernel_shared(CUFFT_COMPLEX * Fimg_D,
                                                      DOUBLE* fweight_D,
                                                      const  CUFFT_COMPLEX * __restrict__ Fimg_shift_nomask_D,
                                                      const DOUBLE* __restrict__ Minvsigma2_D,
                                                      const DOUBLE* __restrict__ weight_D,
                                                      const DOUBLE* __restrict__ mctf_D,
                                                      const DOUBLE* __restrict__ exp_Sum_Weigh_particles,
                                                      const int* __restrict__ Significant_list_D,
                                                      int nr_images,
                                                      int image_size,
                                                      int nr_oversampled_rot,
                                                      int nr_trans,
                                                      int nr_oversampled_trans,
                                                      int nr_valid_orients,
                                                      int weight_size_x
                                                     )
{
	int tid = threadIdx.x;
	int  ishift_id, iover_rot, orient_id;
	DOUBLE myctf, weightxinvsigma2;
	DOUBLE real, imag, fweight;
	DOUBLE Minvsigma2;

	int ctf_image_offset;

	iover_rot = blockIdx.x % nr_oversampled_rot;
	orient_id = Significant_list_D[blockIdx.x / nr_oversampled_rot];

	DOUBLE* sum_weight = (DOUBLE*) local_weight_array;
	DOUBLE* weight_array = (DOUBLE*) &sum_weight [2 * nr_images];
	if (tid < 2 * nr_images)
	{
		sum_weight[tid] = exp_Sum_Weigh_particles[tid];
	}
	for (int k = tid; k < nr_images * nr_trans * nr_oversampled_trans; k += blockDim.x)
	{
		int im = k / (nr_trans * nr_oversampled_trans);
		int trans_index = (k % (nr_trans * nr_oversampled_trans)) / (nr_oversampled_trans);
		int over_trans = k % nr_oversampled_trans;
		weight_array[k] = weight_D[over_trans +   trans_index * nr_oversampled_rot * nr_oversampled_trans + im * weight_size_x + orient_id * nr_trans * nr_oversampled_rot * nr_oversampled_trans + iover_rot * nr_oversampled_trans];
	}
	int pix_id = blockIdx.y * blockDim.x + tid;
	if (pix_id >= image_size)
	{
		return;
	}

	real = imag = fweight = 0.;
	__syncthreads();
	for (int im = 0; im < nr_images; im++)
	{
		ctf_image_offset = im * nr_valid_orients * nr_oversampled_rot * image_size + blockIdx.x * image_size;
		myctf = mctf_D[pix_id + ctf_image_offset]; //im*image_size
		Minvsigma2 =  Minvsigma2_D[pix_id + im * image_size];
		ishift_id = im  * nr_oversampled_trans * nr_trans;

		for (int trans_id = 0; trans_id < nr_trans; trans_id++)
		{

			for (int over_rans = 0; over_rans <  nr_oversampled_trans;  over_rans++)
			{
				if (weight_array[im * nr_trans * nr_oversampled_trans + trans_id * nr_oversampled_trans + over_rans] >= sum_weight[im])
				{
					weightxinvsigma2 = (weight_array[im * nr_trans * nr_oversampled_trans + trans_id * nr_oversampled_trans + over_rans] / sum_weight[im + nr_images]) * myctf * Minvsigma2;
					real += Fimg_shift_nomask_D[pix_id + (ishift_id + trans_id * nr_oversampled_trans + over_rans) * image_size].x * weightxinvsigma2;
					imag += Fimg_shift_nomask_D[pix_id + (ishift_id + trans_id * nr_oversampled_trans + over_rans) * image_size].y * weightxinvsigma2;
					fweight += weightxinvsigma2 * myctf;
				}

			}

		}
	}
	Fimg_D[pix_id + blockIdx.x * image_size].x += real;
	Fimg_D[pix_id + blockIdx.x * image_size].y += imag;
	fweight_D[pix_id + blockIdx.x * image_size] += fweight;
}

void calculate_sum_shift_img_shared_gpu(CUFFT_COMPLEX * Fimg_D,
                                        DOUBLE* fweight_D,
                                        CUFFT_COMPLEX * Fimg_shift_nomask_D,
                                        DOUBLE* Minvsigma2_D,
                                        DOUBLE* weight_D,
                                        DOUBLE* mctf_D,
                                        DOUBLE* exp_Sum_Weigh_particles,
                                        int* Significant_list_D,
                                        int image_size,
                                        int nr_images,
                                        int nr_oversampled_rot,
                                        int nr_trans,
                                        int nr_oversampled_trans,
                                        int nr_valid_orients,
                                        int exp_Mweight_xdim
                                       )
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	int blk_y = (image_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128;
	dim3 dimGrid(nr_valid_orients * nr_oversampled_rot, blk_y, 1);
	int shared_size = nr_trans * nr_oversampled_trans * nr_images * sizeof(DOUBLE) + nr_images * 2 * sizeof(DOUBLE);
	calculate_sum_shift_img_kernel_shared <<< dimGrid, dimBlock, shared_size>>>(Fimg_D,
	                                                                            fweight_D,
	                                                                            Fimg_shift_nomask_D,
	                                                                            Minvsigma2_D,
	                                                                            weight_D,
	                                                                            mctf_D,
	                                                                            exp_Sum_Weigh_particles,
	                                                                            Significant_list_D,
	                                                                            nr_images,
	                                                                            image_size,
	                                                                            nr_oversampled_rot,
	                                                                            nr_trans,
	                                                                            nr_oversampled_trans,
	                                                                            nr_valid_orients,
	                                                                            exp_Mweight_xdim
	                                                                           );

}
__global__ void calculate_diff2_no_do_squared_difference_kernel(DOUBLE* diff2_D,
                                                                CUFFT_COMPLEX * frefctf_D,
                                                                CUFFT_COMPLEX * Fimg_shift_D,
                                                                DOUBLE* exp_local_sqrtXi2_D,
                                                                int* valid_orient_trans_index_D,
                                                                int* valid_orient_prefix_sum,
                                                                int exp_nr_particles,
                                                                int exp_nr_orients,
                                                                int exp_nr_trans,
                                                                int exp_nr_oversampled_rot,
                                                                int exp_nr_oversampled_trans,
                                                                int nr_valid_orients,
                                                                int diff_xdim,
                                                                int image_size)
{
	int bid_x = blockIdx.x;
	int part_id = valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] / (exp_nr_orients * exp_nr_trans);
	int tid = threadIdx.x;
	int orient_id = (valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] % (exp_nr_orients * exp_nr_trans)) / (exp_nr_trans);
	int trans_id = (valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] % exp_nr_trans);

	__shared__ DOUBLE diff2_array[BLOCK_SIZE], suma2_array[BLOCK_SIZE];
	diff2_array[tid] = 0.;
	suma2_array[tid] = 0.;

	DOUBLE suma2_real = 0., suma2_imag = 0.;

	int oversampled_rot_id =  bid_x % (exp_nr_oversampled_rot * exp_nr_oversampled_trans) / exp_nr_oversampled_trans;
	int oversampled_trans_id =  bid_x % exp_nr_oversampled_trans;
	int shift_id = part_id * exp_nr_oversampled_trans * exp_nr_trans +
	               trans_id * exp_nr_oversampled_trans + oversampled_trans_id;
	int reordered_orient_id = valid_orient_prefix_sum[orient_id];
	int ref_image_offset = (part_id * nr_valid_orients * exp_nr_oversampled_rot + reordered_orient_id * exp_nr_oversampled_rot + oversampled_rot_id) * image_size;

	const  CUFFT_COMPLEX * __restrict__ thisthread_Frefctf_D = &frefctf_D[ref_image_offset]; //(part_id+(reordered_orient_id*exp_nr_oversampled_rot+oversampled_rot_id)*exp_nr_particles)
	const  CUFFT_COMPLEX * __restrict__ thisthread_Fimg_shift_D = &Fimg_shift_D[shift_id * image_size];  //shift_id * image_size


	for (int i = tid; i < image_size; i += BLOCK_SIZE)
	{

		diff2_array[tid] += (thisthread_Frefctf_D[i].x * thisthread_Fimg_shift_D[i].x + thisthread_Frefctf_D[i].y * thisthread_Fimg_shift_D[i].y);

		suma2_real = thisthread_Frefctf_D[i].x;
		suma2_imag = thisthread_Frefctf_D[i].y;
		suma2_array[tid] += suma2_real * suma2_real + suma2_imag * suma2_imag;
	}
	__syncthreads();
	for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			diff2_array[tid] += diff2_array[tid + s];
			suma2_array[tid] += suma2_array[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();

	if (tid == 0)
	{
		diff2_D[part_id * diff_xdim + ((orient_id * exp_nr_trans + trans_id)* exp_nr_oversampled_rot + oversampled_rot_id)*  exp_nr_oversampled_trans + oversampled_trans_id] = - diff2_array[0] / (sqrt(suma2_array[0]) * exp_local_sqrtXi2_D[part_id]);
	}
}

void calculate_diff2_no_do_squared_difference_gpu(DOUBLE* diff2_D,
                                                  CUFFT_COMPLEX * frefctf_D,
                                                  CUFFT_COMPLEX * Fimg_shift_D,
                                                  DOUBLE* exp_local_sqrtXi2_D,
                                                  int* valid_orient_trans_index_D,
                                                  int* valid_orient_prefix_sum,
                                                  int exp_nr_particles,
                                                  int exp_nr_orients,
                                                  int exp_nr_trans,
                                                  int exp_nr_oversampled_rot,
                                                  int exp_nr_oversampled_trans,
                                                  int nr_valid_orients,
                                                  int diff_xdim,
                                                  int image_size,
                                                  int nr_valid_orient_trans)
{
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(exp_nr_oversampled_rot * exp_nr_oversampled_trans * nr_valid_orient_trans, 1, 1);

	calculate_diff2_no_do_squared_difference_kernel <<< dimGrid, dimBlock>>>(diff2_D,
	                                                                         frefctf_D,
	                                                                         Fimg_shift_D,
	                                                                         exp_local_sqrtXi2_D,
	                                                                         valid_orient_trans_index_D,
	                                                                         valid_orient_prefix_sum,
	                                                                         exp_nr_particles,
	                                                                         exp_nr_orients,
	                                                                         exp_nr_trans,
	                                                                         exp_nr_oversampled_rot,
	                                                                         exp_nr_oversampled_trans,
	                                                                         nr_valid_orients,
	                                                                         diff_xdim,
	                                                                         image_size);
}



__global__ void calculate_diff2_do_squared_difference_kernel(DOUBLE* diff2_D,
                                                             const  CUFFT_COMPLEX * __restrict__  frefctf_D,
                                                             const  CUFFT_COMPLEX * __restrict__  Fimg_shift_D,
                                                             DOUBLE* exp_highres_Xi2_imgs_D,
                                                             DOUBLE* Minvsigma2_D,
                                                             int* valid_orient_trans_index_D,
                                                             int* valid_orient_prefix_sum,
                                                             int exp_nr_particles,
                                                             int exp_nr_orients,
                                                             int exp_nr_trans,
                                                             int exp_nr_oversampled_rot,
                                                             int exp_nr_oversampled_trans,
                                                             int nr_valid_orients,
                                                             int diff_xdim,
                                                             int image_size)
{

	int bid_x = blockIdx.x;
	int part_id = valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] / (exp_nr_orients * exp_nr_trans);
	int tid = threadIdx.x;
	int orient_id = (valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] % (exp_nr_orients * exp_nr_trans)) / (exp_nr_trans);
	int trans_id = (valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] % exp_nr_trans);

	int oversampled_rot_id =  bid_x % (exp_nr_oversampled_rot * exp_nr_oversampled_trans) / exp_nr_oversampled_trans;
	int oversampled_trans_id =  bid_x % exp_nr_oversampled_trans;
	int shift_id = part_id * exp_nr_oversampled_trans * exp_nr_trans + trans_id * exp_nr_oversampled_trans + oversampled_trans_id;

	int reordered_orient_id = valid_orient_prefix_sum[orient_id];
	int ref_image_offset = (part_id * nr_valid_orients * exp_nr_oversampled_rot + reordered_orient_id * exp_nr_oversampled_rot + oversampled_rot_id) * image_size;

	__shared__ DOUBLE diff2_array[BLOCK_SIZE_128];
	diff2_array[tid] = 0.;

	const  CUFFT_COMPLEX * __restrict__ thisthread_Frefctf_D = &frefctf_D[ref_image_offset];
	const  CUFFT_COMPLEX * __restrict__ thisthread_Fimg_shift_D = &Fimg_shift_D[shift_id * image_size];

	DOUBLE* thisthread_Minvsigma2_D = &Minvsigma2_D[part_id * image_size];

	DOUBLE diff2_real = 0., diff2_imag = 0.;
	for (int i = tid; i < image_size; i += blockDim.x)
	{
		diff2_real = thisthread_Frefctf_D[i].x - thisthread_Fimg_shift_D[i].x;
		diff2_imag = thisthread_Frefctf_D[i].y - thisthread_Fimg_shift_D[i].y;
		diff2_array[tid] += (diff2_real * diff2_real + diff2_imag * diff2_imag) * 0.5 * thisthread_Minvsigma2_D[i];
	}

	__syncthreads();
	for (int s = BLOCK_SIZE_128 >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			diff2_array[tid] += diff2_array[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		diff2_D[part_id * diff_xdim + ((orient_id * exp_nr_trans + trans_id)* exp_nr_oversampled_rot + oversampled_rot_id)*  exp_nr_oversampled_trans + oversampled_trans_id] = diff2_array[0] + exp_highres_Xi2_imgs_D[part_id] * 0.5;
	}
}


void calculate_diff2_do_squared_difference_gpu(DOUBLE* diff2_D,
                                               CUFFT_COMPLEX * frefctf_D,
                                               CUFFT_COMPLEX * Fimg_shift_D,
                                               DOUBLE* exp_highres_Xi2_imgs_D,
                                               DOUBLE* Minvsigma2_D,
                                               int* valid_orient_trans_index_D,
                                               int* valid_orient_prefix_sum,
                                               int exp_nr_particles,
                                               int exp_nr_orients,
                                               int exp_nr_trans,
                                               int exp_nr_oversampled_rot,
                                               int exp_nr_oversampled_trans,
                                               int nr_valid_orients,
                                               int diff_xdim,
                                               int image_size,
                                               int  nr_valid_orient_trans)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(exp_nr_oversampled_rot * exp_nr_oversampled_trans * nr_valid_orient_trans, 1, 1);
	calculate_diff2_do_squared_difference_kernel <<< dimGrid, dimBlock>>>(diff2_D,
	                                                                      frefctf_D,
	                                                                      Fimg_shift_D,
	                                                                      exp_highres_Xi2_imgs_D,
	                                                                      Minvsigma2_D,
	                                                                      valid_orient_trans_index_D,
	                                                                      valid_orient_prefix_sum,
	                                                                      exp_nr_particles,
	                                                                      exp_nr_orients,
	                                                                      exp_nr_trans,
	                                                                      exp_nr_oversampled_rot,
	                                                                      exp_nr_oversampled_trans,
	                                                                      nr_valid_orients,
	                                                                      diff_xdim,
	                                                                      image_size);
}

__global__ void init_exp_mweight_kernel(DOUBLE* exp_Mweight_D, DOUBLE c, int size)
{
	int tid = threadIdx.x;
	int offset = blockIdx.x * BLOCK_SIZE ;
	int index = tid + offset;
	if (index >= size)
	{
		return;
	}
	exp_Mweight_D[index] = c;
}


void init_exp_mweight_gpu(DOUBLE* exp_Mweight_D, DOUBLE c, int size)
{
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(size / BLOCK_SIZE + 1, 1, 1);
	init_exp_mweight_kernel <<< dimGrid, dimBlock>>>(exp_Mweight_D, c, size);

}

__global__ void init_exp_min_diff2_kernel(DOUBLE* exp_min_diff2_D, DOUBLE c, int size)
{
	int tid = threadIdx.x;
	int offset = blockIdx.x * BLOCK_SIZE ;
	int index = tid + offset;
	if (index >= size)
	{
		return;
	}
	exp_min_diff2_D[index] = c;
}


void init_exp_min_diff2_gpu(DOUBLE* exp_min_diff2_D, DOUBLE c, int size)
{
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(size / BLOCK_SIZE + 1, 1, 1);
	init_exp_mweight_kernel <<< dimGrid, dimBlock>>>(exp_min_diff2_D, c, size);
}

__global__ void calculate_weight_kernel(const int nr_particles,
                                        DOUBLE* exp_sum_weight_D,
                                        int* exp_none_zero_number,
                                        DOUBLE* max_weight,
                                        DOUBLE* exp_Mweight_D,
                                        const  DOUBLE* __restrict__ exp_min_diff2_D,
                                        const  DOUBLE* __restrict__ pdf_orientation_D,
                                        const  DOUBLE* __restrict__ pdf_offset_D,
                                        long int xdim_Mweight,
                                        int iclass_min,
                                        int iclass_max,
                                        int model_nr_classes,
                                        int exp_nr_classes,
                                        long int nr_elements,
                                        long int nr_orients,
                                        long int exp_nr_trans,
                                        long int exp_nr_oversampled_rot,
                                        long int exp_nr_oversampled_trans)
{
	int bid_x = blockIdx.x;
	int ipart  = blockIdx.y;
	int tid = threadIdx.x;
	int index = bid_x * blockDim.x + tid;
	__shared__ DOUBLE local_sum_weight[BLOCK_SIZE_128];
	__shared__ int none_zeor_weight[BLOCK_SIZE_128];
	local_sum_weight[tid] = 0.;
	none_zeor_weight[tid] = 0;

	if (index >= ((iclass_max - iclass_min + 1) * nr_orients * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans))
	{
		return;
	}

	int iclass = index / (nr_orients * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans) + iclass_min;
	int iorient = index / (exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans) % nr_orients;
	int itrans = index / (exp_nr_oversampled_rot * exp_nr_oversampled_trans) % exp_nr_trans;

	DOUBLE weight = pdf_orientation_D[(ipart * model_nr_classes + (iclass)) * nr_orients + iorient] * pdf_offset_D[(ipart * model_nr_classes + (iclass)) * exp_nr_trans + itrans];
	DOUBLE* thisthread_exp_Mweight_D = &exp_Mweight_D[ipart * xdim_Mweight + iclass_min * nr_orients * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans];

	if (thisthread_exp_Mweight_D[index] < 0.)
	{
		thisthread_exp_Mweight_D[index] = 0.;
	}
	else
	{
		DOUBLE diff2 = thisthread_exp_Mweight_D[index] - exp_min_diff2_D[ipart];
		if (diff2 > 700.)
		{
			thisthread_exp_Mweight_D[index] = 0.;
		}
		else
		{
			thisthread_exp_Mweight_D[index] = weight * exp(-diff2);
			local_sum_weight[tid] = weight * exp(-diff2);
			none_zeor_weight[tid] = 1;
		}
	}

	//calculate the partial sum of each thread block
	__syncthreads();
	for (int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			local_sum_weight[tid] += local_sum_weight[tid + s];
			none_zeor_weight[tid] += none_zeor_weight[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0)
	{
		exp_sum_weight_D[blockIdx.x + blockIdx.y * gridDim.x] = local_sum_weight[0];
		exp_none_zero_number[blockIdx.x + blockIdx.y * gridDim.x] = none_zeor_weight[0];
	}

}

void calculate_weight_gpu(const int nr_particles,
                          DOUBLE* exp_sum_weight_D,
                          int* exp_none_zero_number,
                          DOUBLE* max_weight,
                          DOUBLE* exp_Mweight_D,
                          DOUBLE* exp_min_diff2_D,
                          DOUBLE* pdf_orientation_D,
                          DOUBLE* pdf_offset_D,
                          long int xdim_Mweight,
                          int iclass_min,
                          int iclass_max,
                          int model_nr_classes,
                          int exp_nr_classes,
                          long int nr_elements,
                          long int nr_orients,
                          long int exp_nr_trans,
                          long int exp_nr_oversampled_rot,
                          long int exp_nr_oversampled_trans)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((((iclass_max - iclass_min + 1) * nr_orients * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans) + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128 , nr_particles, 1);
	calculate_weight_kernel <<< dimGrid, dimBlock>>>(nr_particles,
	                                                 exp_sum_weight_D,
	                                                 exp_none_zero_number,
	                                                 max_weight,
	                                                 exp_Mweight_D,
	                                                 exp_min_diff2_D,
	                                                 pdf_orientation_D,
	                                                 pdf_offset_D,
	                                                 xdim_Mweight,
	                                                 iclass_min,
	                                                 iclass_max,
	                                                 model_nr_classes,
	                                                 exp_nr_classes,
	                                                 nr_elements,
	                                                 nr_orients,
	                                                 exp_nr_trans,
	                                                 exp_nr_oversampled_rot,
	                                                 exp_nr_oversampled_trans);
}


__global__ void calculate_weight_first_iter_pass0_kernel(
    DOUBLE* exp_Mweight_D,
    bool* exp_Mcoarse_significant_D,
    const  DOUBLE* __restrict__ exp_min_diff2_D,
    long int xdim_Mweight,
    int iclass_min,
    int iclass_max,
    long int nr_elements
)
{
	int bid_x = blockIdx.x;
	int ipart  = blockIdx.y;
	int tid = threadIdx.x;
	int index = bid_x * blockDim.x + tid;

	if (index >= ((iclass_max - iclass_min + 1) * nr_elements))
	{
		return;
	}

	DOUBLE* thisthread_exp_Mweight_D = &exp_Mweight_D[ipart * xdim_Mweight + iclass_min * nr_elements];
	bool* thisthread_exp_Mcoarse_significant_D = &exp_Mcoarse_significant_D[ipart * xdim_Mweight + iclass_min * nr_elements];
	if (thisthread_exp_Mweight_D[index]  == exp_min_diff2_D[ipart])
	{
		thisthread_exp_Mweight_D[index] = 1.;
		thisthread_exp_Mcoarse_significant_D[index] = true;
	}
	else
	{
		thisthread_exp_Mweight_D[index] = 0.;
		thisthread_exp_Mcoarse_significant_D[index] = false;

	}

}
__global__ void calculate_weight_first_iter_pass1_kernel(
    DOUBLE* exp_Mweight_D,
    const  DOUBLE* __restrict__ exp_min_diff2_D,
    long int xdim_Mweight,
    int iclass_min,
    int iclass_max,
    long int nr_elements
)
{
	int bid_x = blockIdx.x;
	int ipart  = blockIdx.y;
	int tid = threadIdx.x;
	int index = bid_x * blockDim.x + tid;

	if (index >= ((iclass_max - iclass_min + 1) * nr_elements))
	{
		return;
	}

	DOUBLE* thisthread_exp_Mweight_D = &exp_Mweight_D[ipart * xdim_Mweight + iclass_min * nr_elements];
	if (thisthread_exp_Mweight_D[index]  == exp_min_diff2_D[ipart])
	{
		thisthread_exp_Mweight_D[index] = 1.;
	}
	else
	{
		thisthread_exp_Mweight_D[index] = 0.;

	}

}
void calculate_weight_first_iter_gpu(DOUBLE* exp_Mweight_D,
                                     bool* exp_Mcoarse_significant_D,
                                     DOUBLE* exp_min_diff2_D,
                                     const int nr_particles,
                                     long int xdim_Mweight,
                                     int iclass_min,
                                     int iclass_max,
                                     long int nr_elements,
                                     int exp_ipass)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((((iclass_max - iclass_min + 1) * nr_elements) + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128 , nr_particles, 1);
	if (exp_ipass == 0)
	{
		calculate_weight_first_iter_pass0_kernel <<< dimGrid, dimBlock>>>(exp_Mweight_D,
		                                                                  exp_Mcoarse_significant_D,
		                                                                  exp_min_diff2_D,
		                                                                  xdim_Mweight,
		                                                                  iclass_min,
		                                                                  iclass_max,
		                                                                  nr_elements);
	}
	else
	{
		calculate_weight_first_iter_pass1_kernel <<< dimGrid, dimBlock>>>(exp_Mweight_D,
		                                                                  exp_min_diff2_D,
		                                                                  xdim_Mweight,
		                                                                  iclass_min,
		                                                                  iclass_max,
		                                                                  nr_elements);
	}
}


__global__ void calculate_A_2d_kernel(CUFFT_COMPLEX * Fref_all_dev,
                                      const DOUBLE*  __restrict__ A_D,
                                      const CUFFT_COMPLEX * __restrict__ data_D,
                                      bool inv,
                                      int padding_factor,
                                      int r_max,
                                      int r_min_nn,
                                      int f2d_x,
                                      int f2d_y,
                                      int data_x,
                                      int data_y,
                                      int data_z,
                                      int data_starty,
                                      int data_startz,
                                      int nr_A)
{

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int my_r_max;
	if (r_max < f2d_x - 1)
	{
		my_r_max = r_max + 1;
	}
	else
	{
		my_r_max = f2d_x;
	}
	int max_tid = f2d_y * my_r_max;
	int max_r2 = (my_r_max - 1) * (my_r_max - 1);

	DOUBLE fx, fy, xp, yp;
	int x0, x1, y0, y1, y, r2;
	bool is_neg_x;
	CUFFT_COMPLEX  d00, d01, d10, d11, dx0, dx1;

	int i, x;
	__shared__ DOUBLE Ainv[3 * 3];

	// f2d should already be in the right size (ori_size,orihalfdim)
	// AND the points outside max_r should already be zero...
	if (tid < 3 * 3)
	{
		if (inv)
		{
			Ainv[tid] = A_D[tid + blockIdx.x * 9] * (DOUBLE)padding_factor;
		}
		else
		{
			int inv_row = tid / 3;
			int inv_vol = tid % 3;
			Ainv[tid] = A_D[inv_vol * 3 + inv_row + blockIdx.x * 9] * (DOUBLE)padding_factor;
		}
	}
	__syncthreads();
	for (int k = tid; k < max_tid; k += BLOCK_SIZE)
	{
		i = k / my_r_max;
		x = k % my_r_max;
		if (i < my_r_max)
		{
			y = i;
		}
		else if (i > f2d_y - my_r_max)
		{
			y = i - f2d_y;
		}
		else
		{
			continue;
		}
		r2 = x * x + y * y;
		if (r2 > max_r2)
		{
			continue;
		}
		xp = Ainv[0] * x + Ainv[1] * y;
		yp = Ainv[3] * x + Ainv[4] * y;

		if (xp < 0)
		{
			xp = -xp;
			yp = -yp;
			is_neg_x = true;
		}
		else
		{
			is_neg_x = false;
		}

		x0 = floor(xp);
		fx = xp - x0;
		x1 = x0 + 1;

		y0 = floor(yp);
		if (y0 > yp)
		{
			y0--;
		}
		fy = yp - y0;
		y0 -= data_starty;
		y1 = y0 + 1;

		d00 = data_D[y0 * data_x + x0];
		d01 = data_D[y0 * data_x + x1];
		d10 = data_D[y1 * data_x + x0];
		d11 = data_D[y1 * data_x + x1];


		dx0.x = d00.x + (d01.x - d00.x) * fx;
		dx0.y = d00.y + (d01.y - d00.y) * fx;

		dx1.x = d10.x + (d11.x - d10.x) * fx;
		dx1.y = d10.y + (d11.y - d10.y) * fx;


		if (is_neg_x)
		{
			Fref_all_dev[bid * f2d_x * f2d_y + i * f2d_x + x].x =   dx0.x + (dx1.x - dx0.x) * fy;
			Fref_all_dev[bid * f2d_x * f2d_y + i * f2d_x + x].y = - dx0.y + (dx1.y - dx0.y) * fy;
		}
		else
		{
			Fref_all_dev[bid * f2d_x * f2d_y + i * f2d_x + x].x =  dx0.x + (dx1.x - dx0.x) * fy;
			Fref_all_dev[bid * f2d_x * f2d_y + i * f2d_x + x].y =  dx0.y + (dx1.y - dx0.y) * fy;
		}
	}
}
__global__ void calculate_A_3d_kernel(CUFFT_COMPLEX * Fref_all_D,
                                      const DOUBLE*  __restrict__ A_D,
                                      const CUFFT_COMPLEX * __restrict__ data_D,
                                      bool inv,
                                      int padding_factor,
                                      int r_max,
                                      int r_min_nn,
                                      int f2d_x,
                                      int f2d_y,
                                      int data_x,
                                      int data_y,
                                      int data_z,
                                      int data_starty,
                                      int data_startz,
                                      int nr_A)
{

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int my_r_max;
	if (r_max < f2d_x - 1)
	{
		my_r_max = r_max + 1;
	}
	else
	{
		my_r_max = f2d_x;
	}
	int max_tid = f2d_y * my_r_max;
	int max_r2 = (my_r_max - 1) * (my_r_max - 1);

	DOUBLE fx, fy, fz, xp, yp, zp;
	int x0, x1, y0, y1, z0, z1, y, r2;
	bool is_neg_x;
	CUFFT_COMPLEX  d000, d001, d010, d011, d100, d101, d110, d111;
	CUFFT_COMPLEX  dx00, dx01, dx10, dx11, dxy0, dxy1;

	int i, x;
	__shared__ DOUBLE Ainv[3 * 3];

	// f2d should already be in the right size (ori_size,orihalfdim)
	// AND the points outside max_r should already be zero...
	if (tid < 3 * 3)
	{
		if (inv)
		{
			Ainv[tid] = A_D[tid + blockIdx.x * 9] * (DOUBLE)padding_factor;
		}
		else
		{
			int inv_row = tid / 3;
			int inv_vol = tid % 3;
			Ainv[tid] = A_D[inv_vol * 3+ inv_row + blockIdx.x * 9] * (DOUBLE)padding_factor;
		}
	}
	__syncthreads();
	for (int k = tid; k < max_tid; k += BLOCK_SIZE)
	{
		i = k / my_r_max;
		x = k % my_r_max;
		if (i < my_r_max)
		{
			y = i;
		}
		else if (i > f2d_y - my_r_max)
		{
			y = i - f2d_y;
		}
		else
		{
			continue;
		}
		r2 = x * x + y * y;
		if (r2 > max_r2)
		{
			continue;
		}
		xp = Ainv[0] * x + Ainv[1] * y;
		yp = Ainv[3] * x + Ainv[4] * y;
		zp = Ainv[6] * x + Ainv[7] * y;

		if (xp < 0)
		{
			xp = -xp;
			yp = -yp;
			zp = -zp;
			is_neg_x = true;
		}
		else
		{
			is_neg_x = false;
		}

		x0 = floor(xp);
		fx = xp - x0;
		x1 = x0 + 1;

		y0 = floor(yp);
		if (y0 > yp)
		{
			y0--;
		}
		fy = yp - y0;
		y0 -= data_starty;
		y1 = y0 + 1;

		z0 = floor(zp);
		if (z0 > zp)
		{
			z0--;
		}
		fz = zp - z0;
		z0 -= data_startz;
		z1 = z0 + 1;

		d000 = data_D[z0 * data_y * data_x + y0 * data_x + x0];
		d001 = data_D[z0 * data_y * data_x + y0 * data_x + x1];
		d010 = data_D[z0 * data_y * data_x + y1 * data_x + x0];
		d011 = data_D[z0 * data_y * data_x + y1 * data_x + x1];
		d100 = data_D[z1 * data_y * data_x + y0 * data_x + x0];
		d101 = data_D[z1 * data_y * data_x + y0 * data_x + x1];
		d110 = data_D[z1 * data_y * data_x + y1 * data_x + x0];
		d111 = data_D[z1 * data_y * data_x + y1 * data_x + x1];


		dx00.x = d000.x + (d001.x - d000.x) * fx;
		dx00.y = d000.y + (d001.y - d000.y) * fx;
		dx01.x = d100.x + (d101.x - d100.x) * fx;
		dx01.y = d100.y + (d101.y - d100.y) * fx;
		dx10.x = d010.x + (d011.x - d010.x) * fx;
		dx10.y = d010.y + (d011.y - d010.y) * fx;
		dx11.x = d110.x + (d111.x - d110.x) * fx;
		dx11.y = d110.y + (d111.y - d110.y) * fx;
		dxy0.x = dx00.x + (dx10.x - dx00.x) * fy;
		dxy0.y = dx00.y + (dx10.y - dx00.y) * fy;
		dxy1.x = dx01.x + (dx11.x - dx01.x) * fy;
		dxy1.y = dx01.y + (dx11.y - dx01.y) * fy;



		if (is_neg_x)
		{
			Fref_all_D[bid * f2d_x * f2d_y + i * f2d_x + x].x =   dxy0.x + (dxy1.x - dxy0.x) * fz;
			Fref_all_D[bid * f2d_x * f2d_y + i * f2d_x + x].y = - dxy0.y - (dxy1.y - dxy0.y) * fz;
		}
		else
		{
			Fref_all_D[bid * f2d_x * f2d_y + i * f2d_x + x].x = dxy0.x + (dxy1.x - dxy0.x) * fz;
			Fref_all_D[bid * f2d_x * f2d_y + i * f2d_x + x].y = dxy0.y + (dxy1.y - dxy0.y) * fz;
		}
	}
}

void get2DFourierTransform_gpu(CUFFT_COMPLEX * Fref_all_D, DOUBLE* A_D, CUFFT_COMPLEX * data_D, bool inv, int padding_factor, int r_max, int r_min_nn, int f2d_x, int f2d_y, int data_x, int data_y, int data_z, int data_starty, int data_startz, int nr_A, int ref_dim)
{
	if (nr_A > 0)
	{
		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		dim3 dimGrid(nr_A, 1, 1);
		if (ref_dim == 2)
		{
			calculate_A_2d_kernel <<< dimGrid, dimBlock>>>(Fref_all_D, A_D, data_D, inv, padding_factor,  r_max, r_min_nn, f2d_x, f2d_y, data_x, data_y, data_z, data_starty, data_startz, nr_A);
		}
		else
		{
			//std::cout<<"Doing 3D  FourierTransform" << data_x << "  " << data_y << "  " << data_z<< std::endl;
			calculate_A_3d_kernel <<< dimGrid, dimBlock>>>(Fref_all_D, A_D, data_D, inv, padding_factor,  r_max, r_min_nn, f2d_x, f2d_y, data_x, data_y, data_z, data_starty, data_startz, nr_A);
		}
	}
}


__global__ void  backrotate2D_kernel(CUFFT_COMPLEX * f2d_D,
                                     DOUBLE* A_D,
                                     bool inv,
                                     DOUBLE* Mweight_D,
                                     CUFFT_COMPLEX * data_D,
                                     DOUBLE* weight_D,
                                     int padding_factor,
                                     int r_max,
                                     int max_r2_D,
                                     int min_r2_nn_D,
                                     int ydim_f2d,
                                     int xdim_f2d,
                                     int image_size,
                                     int interpolator,
                                     int start_y,
                                     int out_XDIM
                                    )
{
	DOUBLE fx, fy, mfx, mfy, xp, yp;
	int first_x, x0, x1, y0, y1, y, y2, r2;
	bool is_neg_x;
	DOUBLE dd000, dd001, dd010, dd011;
	CUFFT_COMPLEX   my_val;
	__shared__ DOUBLE Ainv[3 * 3];
	DOUBLE my_weight = 1.;
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid = tid_x + tid_y * blockDim.x;

	// f2d should already be in the right size (ori_size,orihalfdim)
	// AND the points outside max_r should already be zero...
	if (tid < 3 * 3)
	{
		if (inv)
		{
			Ainv[tid] = A_D[tid + blockIdx.x * 9] * (DOUBLE)padding_factor;
		}
		else
		{
			int inv_row = tid / 3;
			int inv_vol = tid % 3;
			Ainv[tid] = A_D[inv_vol * 3 + inv_row + blockIdx.x * 9] * (DOUBLE)padding_factor;
		}
	}
	__syncthreads();
	// Go from the 2D slice coordinates to the 3D coordinates

	for (int i = tid_y; i < ydim_f2d; i += blockDim.y)
	{
		// Dont search beyond square with side max_r
		if (i <= r_max)
		{
			y = i;
			first_x = 0;
		}
		else if (i >= (ydim_f2d - r_max))
		{
			y = i - ydim_f2d;
			// x==0 plane is stored twice in the FFTW format. Dont set it twice in BACKPROJECTION!
			first_x = 1;
		}
		else
		{
			continue;
		}

		y2 = y * y;
		for (int x = (tid_x + first_x); x <= r_max; x += blockDim.x)
		{
			// Only include points with radius < max_r (exclude points outside circle in square)
			r2 = x * x + y2;
			if (r2 > max_r2_D)
			{
				continue;
			}

			// Get the relevant value in the input image
			my_val = f2d_D[blockIdx.x * image_size + i * xdim_f2d + x]; //DIRECT_A2D_ELEM(f2d, i, x);

			// Get the weight
			if (Mweight_D != NULL)
			{
				my_weight = Mweight_D[blockIdx.x * image_size + i * xdim_f2d + x];    //DIRECT_A2D_ELEM(*Mweight, i, x);
			}
			// else: my_weight was already initialised to 1.

			if (my_weight > 0.)
			{

				// Get logical coordinates in the 3D map
				xp = Ainv[0 * 3 + 0] * x + Ainv[0 * 3 + 1] * y;
				yp = Ainv[1 * 3 + 0] * x + Ainv[1 * 3 + 1] * y;
				if (interpolator == 1 || r2 < min_r2_nn_D) //TRILINEAR 1
				{
					// Only asymmetric half is stored
					if (xp < 0)
					{
						// Get complex conjugated hermitian symmetry pair
						xp = -xp;
						yp = -yp;
						is_neg_x = true;
					}
					else
					{
						is_neg_x = false;
					}

					// Trilinear interpolation (with physical coords)
					// Subtract STARTINGY and STARTINGZ to accelerate access to data (STARTINGX=0)
					// In that way use DIRECT_A3D_ELEM, rather than A3D_ELEM
					x0 = floor(xp);//(((xp) == (int)(xp)) ? (int)(xp):(((xp) > 0) ? (int)(xp) :   (int)((xp) - 1)));
					fx = xp - x0;
					x1 = x0 + 1;

					y0 = floor(yp);
					fy = yp - y0;
					y0 -=  start_y;//STARTINGY(data);
					y1 = y0 + 1;

					mfx = 1. - fx;
					mfy = 1. - fy;

					if (is_neg_x)
					{
						my_val.y = -my_val.y;
					}

					dd000 = mfy * mfx;
					dd001 = mfy *  fx;
					dd010 = fy * mfx;
					dd011 = fy *  fx;

					atomicAdd(&(data_D[y0 * out_XDIM + x0].x), (DOUBLE)(dd000 * my_val.x));
					atomicAdd(&(data_D[y0 * out_XDIM + x0].y), (DOUBLE)(dd000 * my_val.y));
					atomicAdd(&(data_D[y0 * out_XDIM + x1].x), (DOUBLE)(dd001 * my_val.x));
					atomicAdd(&(data_D[y0 * out_XDIM + x1].y), (DOUBLE)(dd001 * my_val.y));

					atomicAdd(&(data_D[y1 * out_XDIM + x0].x), (DOUBLE)(dd010 * my_val.x));
					atomicAdd(&(data_D[y1 * out_XDIM + x0].y), (DOUBLE)(dd010 * my_val.y));
					atomicAdd(&(data_D[y1 * out_XDIM + x1].x), (DOUBLE)(dd011 * my_val.x));
					atomicAdd(&(data_D[y1 * out_XDIM + x1].y), (DOUBLE)(dd011 * my_val.y));

					atomicAdd(&(weight_D[y0 * out_XDIM + x0]), (DOUBLE)(dd000 * my_weight));
					atomicAdd(&(weight_D[y0 * out_XDIM + x1]), (DOUBLE)(dd001 * my_weight));

					atomicAdd(&(weight_D[y1 * out_XDIM + x0]), (DOUBLE)(dd010 * my_weight));
					atomicAdd(&(weight_D[y1 * out_XDIM + x1]), (DOUBLE)(dd011 * my_weight));

				} // endif TRILINEAR
				else if (interpolator == 0)  //NEAREST_NEIGHBOUR 0
				{

					x0 = lround(xp);
					y0 = lround(yp);

					if (x0 < 0)
					{
						atomicAdd(&(data_D[-y0 * out_XDIM - x0].x), (DOUBLE) my_val.x);
						atomicAdd(&(data_D[-y0 * out_XDIM - x0].y), (DOUBLE)(-my_val.y));
						atomicAdd(&(weight_D[-y0 * out_XDIM - x0]), (DOUBLE) my_weight);

					}
					else
					{
						atomicAdd(&(data_D[y0 * out_XDIM + x0].x), (DOUBLE) my_val.x);
						atomicAdd(&(data_D[y0 * out_XDIM + x0].y), (DOUBLE) my_val.y);
						atomicAdd(&(weight_D[y0 * out_XDIM + x0]), (DOUBLE) my_weight);
					}

				} // endif NEAREST_NEIGHBOUR

			} // endif weight>0.
		} // endif x-loop
	} // endif y-loop
}

__global__ void  backproject_kernel(const CUFFT_COMPLEX * __restrict__ f2d_D,
                                    const DOUBLE* __restrict__ A_D,
                                    bool inv,
                                    const DOUBLE* __restrict__ Mweight_D,
                                    CUFFT_COMPLEX * data_D,
                                    DOUBLE* weight_D,
                                    int padding_factor,
                                    int r_max,
                                    int max_r2_D,
                                    int min_r2_nn_D,
                                    int ydim_f2d,
                                    int xdim_f2d,
                                    int image_size,
                                    int interpolator,
                                    int start_y,
                                    int start_z,
                                    int out_YXDIM,
                                    int out_XDIM
                                   )
{
	DOUBLE fx, fy, fz, mfx, mfy, mfz, xp, yp, zp;
	int first_x, x0, x1, y0, y1, z0, z1, y, y2, r2;
	bool is_neg_x;
	DOUBLE dd000, dd001, dd010, dd011, dd100, dd101, dd110, dd111;
	CUFFT_COMPLEX   my_val;
	__shared__ DOUBLE Ainv[3 * 3];
	DOUBLE my_weight = 1.;
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid = tid_x + tid_y * blockDim.x;

	// f2d should already be in the right size (ori_size,orihalfdim)
	// AND the points outside max_r should already be zero...
	if (tid < 3 * 3)
	{
		if (inv)
		{
			Ainv[tid] = A_D[tid + blockIdx.x * 9] * (DOUBLE)padding_factor;
		}
		else
		{
			int inv_row = tid / 3;
			int inv_vol = tid % 3;
			Ainv[tid] = A_D[inv_vol * 3 + inv_row + blockIdx.x * 9] * (DOUBLE)padding_factor;
		}
	}
	__syncthreads();
	// Go from the 2D slice coordinates to the 3D coordinates

	for (int i = tid_y; i < ydim_f2d; i += blockDim.y)
	{

		// Dont search beyond square with side max_r
		if (i <= r_max)
		{
			y = i;
			first_x = 0;
		}
		else if (i >= (ydim_f2d - r_max))
		{
			y = i - ydim_f2d;
			first_x = 1;
		}
		else
		{
			continue;
		}

		y2 = y * y;
		for (int x = (tid_x + first_x); x <= r_max; x += blockDim.x)
		{
			// Only include points with radius < max_r (exclude points outside circle in square)
			r2 = x * x + y2;
			if (r2 > max_r2_D)
			{
				continue;
			}

			// Get the relevant value in the input image
			my_val = f2d_D[blockIdx.x * image_size + i * xdim_f2d + x]; //DIRECT_A2D_ELEM(f2d, i, x);

			// Get the weight
			if (Mweight_D != NULL)
			{
				my_weight = Mweight_D[blockIdx.x * image_size + i * xdim_f2d + x];    //DIRECT_A2D_ELEM(*Mweight, i, x);
			}
			// else: my_weight was already initialised to 1.

			if (my_weight > 0.)
			{

				// Get logical coordinates in the 3D map
				xp = Ainv[0 * 3 + 0] * x + Ainv[0 * 3 + 1] * y;
				yp = Ainv[1 * 3 + 0] * x + Ainv[1 * 3 + 1] * y;
				zp = Ainv[2 * 3 + 0] * x + Ainv[2 * 3 + 1] * y;

				if (interpolator == 1 || r2 < min_r2_nn_D) //TRILINEAR 1
				{
					// Only asymmetric half is stored
					if (xp < 0)
					{
						// Get complex conjugated hermitian symmetry pair
						xp = -xp;
						yp = -yp;
						zp = -zp;
						is_neg_x = true;
					}
					else
					{
						is_neg_x = false;
					}

					// Trilinear interpolation (with physical coords)
					// Subtract STARTINGY and STARTINGZ to accelerate access to data (STARTINGX=0)
					// In that way use DIRECT_A3D_ELEM, rather than A3D_ELEM
					x0 = floor(xp);//(((xp) == (int)(xp)) ? (int)(xp):(((xp) > 0) ? (int)(xp) :   (int)((xp) - 1)));
					fx = xp - x0;
					x1 = x0 + 1;

					y0 = floor(yp);
					fy = yp - y0;
					y0 -=  start_y;//STARTINGY(data);
					y1 = y0 + 1;

					z0 = floor(zp);
					fz = zp - z0;
					z0 -= start_z;//STARTINGZ(data);
					z1 = z0 + 1;

					mfx = 1. - fx;
					mfy = 1. - fy;
					mfz = 1. - fz;
					if (is_neg_x)
					{
						my_val.y = -my_val.y;
					}

					dd000 = mfz * mfy * mfx;
					dd001 = mfz * mfy *  fx;
					dd010 = mfz *  fy * mfx;
					dd011 = mfz *  fy *  fx;
					dd100 =  fz * mfy * mfx;
					dd101 =  fz * mfy *  fx;
					dd110 =  fz *  fy * mfx;
					dd111 =  fz *  fy *  fx;

					atomicAdd(&(data_D[z0 * out_YXDIM + y0 * out_XDIM + x0].x), (DOUBLE)(dd000 * my_val.x));
					atomicAdd(&(data_D[z0 * out_YXDIM + y0 * out_XDIM + x0].y), (DOUBLE)(dd000 * my_val.y));
					atomicAdd(&(data_D[z0 * out_YXDIM + y0 * out_XDIM + x1].x), (DOUBLE)(dd001 * my_val.x));
					atomicAdd(&(data_D[z0 * out_YXDIM + y0 * out_XDIM + x1].y), (DOUBLE)(dd001 * my_val.y));

					atomicAdd(&(data_D[z0 * out_YXDIM + y1 * out_XDIM + x0].x), (DOUBLE)(dd010 * my_val.x));
					atomicAdd(&(data_D[z0 * out_YXDIM + y1 * out_XDIM + x0].y), (DOUBLE)(dd010 * my_val.y));
					atomicAdd(&(data_D[z0 * out_YXDIM + y1 * out_XDIM + x1].x), (DOUBLE)(dd011 * my_val.x));
					atomicAdd(&(data_D[z0 * out_YXDIM + y1 * out_XDIM + x1].y), (DOUBLE)(dd011 * my_val.y));

					atomicAdd(&(data_D[z1 * out_YXDIM + y0 * out_XDIM + x0].x), (DOUBLE)(dd100 * my_val.x));
					atomicAdd(&(data_D[z1 * out_YXDIM + y0 * out_XDIM + x0].y), (DOUBLE)(dd100 * my_val.y));
					atomicAdd(&(data_D[z1 * out_YXDIM + y0 * out_XDIM + x1].x), (DOUBLE)(dd101 * my_val.x));
					atomicAdd(&(data_D[z1 * out_YXDIM + y0 * out_XDIM + x1].y), (DOUBLE)(dd101 * my_val.y));

					atomicAdd(&(data_D[z1 * out_YXDIM + y1 * out_XDIM + x0].x), (DOUBLE)(dd110 * my_val.x));
					atomicAdd(&(data_D[z1 * out_YXDIM + y1 * out_XDIM + x0].y), (DOUBLE)(dd110 * my_val.y));
					atomicAdd(&(data_D[z1 * out_YXDIM + y1 * out_XDIM + x1].x), (DOUBLE)(dd111 * my_val.x));
					atomicAdd(&(data_D[z1 * out_YXDIM + y1 * out_XDIM + x1].y), (DOUBLE)(dd111 * my_val.y));

					atomicAdd(&(weight_D[z0 * out_YXDIM + y0 * out_XDIM + x0]), (DOUBLE)(dd000 * my_weight));
					atomicAdd(&(weight_D[z0 * out_YXDIM + y0 * out_XDIM + x1]), (DOUBLE)(dd001 * my_weight));

					atomicAdd(&(weight_D[z0 * out_YXDIM + y1 * out_XDIM + x0]), (DOUBLE)(dd010 * my_weight));
					atomicAdd(&(weight_D[z0 * out_YXDIM + y1 * out_XDIM + x1]), (DOUBLE)(dd011 * my_weight));

					atomicAdd(&(weight_D[z1 * out_YXDIM + y0 * out_XDIM + x0]), (DOUBLE)(dd100 * my_weight));
					atomicAdd(&(weight_D[z1 * out_YXDIM + y0 * out_XDIM + x1]), (DOUBLE)(dd101 * my_weight));

					atomicAdd(&(weight_D[z1 * out_YXDIM + y1 * out_XDIM + x0]), (DOUBLE)(dd110 * my_weight));
					atomicAdd(&(weight_D[z1 * out_YXDIM + y1 * out_XDIM + x1]), (DOUBLE)(dd111 * my_weight));

				} // endif TRILINEAR
				else if (interpolator == 0)  //NEAREST_NEIGHBOUR 0
				{

					x0 = lround(xp);
					y0 = lround(yp);
					z0 = lround(zp);

					if (x0 < 0)
					{
						atomicAdd(&(data_D[-z0 * out_YXDIM - y0 * out_XDIM - x0].x), (DOUBLE) my_val.x);
						atomicAdd(&(data_D[-z0 * out_YXDIM - y0 * out_XDIM - x0].y), (DOUBLE)(-my_val.y));
						atomicAdd(&(weight_D[-z0 * out_YXDIM - y0 * out_XDIM - x0]), (DOUBLE) my_weight);

					}
					else
					{
						atomicAdd(&(data_D[z0 * out_YXDIM + y0 * out_XDIM + x0].x), (DOUBLE) my_val.x);
						atomicAdd(&(data_D[z0 * out_YXDIM + y0 * out_XDIM + x0].y), (DOUBLE) my_val.y);
						atomicAdd(&(weight_D[z0 * out_YXDIM + y0 * out_XDIM + x0]), (DOUBLE) my_weight);

					}

				} // endif NEAREST_NEIGHBOUR
			} // endif weight>0.
		} // endif x-loop
	}// endif y-loop
}
void backproject_gpu(CUFFT_COMPLEX * f2d_D,
                     DOUBLE* A_D,
                     bool inv,
                     DOUBLE* Mweight_D,
                     CUFFT_COMPLEX * data_D,
                     DOUBLE* weight_D,
                     DOUBLE padding_factor,
                     int r_max,
                     int r_min_nn,
                     int ydim_f2d,
                     int xdim_f2d,
                     int nr_A,
                     int nr_oversampled_rot,
                     int image_size,
                     int interpolator,
                     int start_y,
                     int start_z,
                     int data_YXDIM,
                     int data_XDIM,
                     int ref_dim
                    )

{
	int min_r2_nn_D = r_min_nn * r_min_nn;
	int max_r2_D = r_max * r_max;
	dim3 dimBlock(BLOCK_X, BLOCK_Y, 1);
	dim3 dimGrid(nr_A, 1, 1);
	if (ref_dim == 3)
	{
		backproject_kernel <<< dimGrid, dimBlock >>>(f2d_D,
		                                             A_D,
		                                             inv,
		                                             Mweight_D,
		                                             data_D,
		                                             weight_D,
		                                             padding_factor,
		                                             r_max,
		                                             max_r2_D,
		                                             min_r2_nn_D,
		                                             ydim_f2d,
		                                             xdim_f2d,
		                                             image_size,
		                                             interpolator,
		                                             start_y,
		                                             start_z,
		                                             data_YXDIM,
		                                             data_XDIM
		                                            );
	}
	else
	{
		backrotate2D_kernel <<< dimGrid, dimBlock >>>(f2d_D,
		                                              A_D,
		                                              inv,
		                                              Mweight_D,
		                                              data_D,
		                                              weight_D,
		                                              padding_factor,
		                                              r_max,
		                                              max_r2_D,
		                                              min_r2_nn_D,
		                                              ydim_f2d,
		                                              xdim_f2d,
		                                              image_size,
		                                              interpolator,
		                                              start_y,
		                                              data_XDIM
		                                             );

	}

}

__global__ void sign_Weight_kernel(const DOUBLE* __restrict__ exp_Mweight_D,
                                   const DOUBLE* __restrict__ exp_Sum_Weigh_particles,
                                   const int* __restrict__  isSignificant_D,
                                   int* valid_weight_list_D,
                                   int exp_Mweight_xdim,
                                   int nr_images,
                                   int nr_orients,
                                   int nr_oversampled_rot,
                                   int nr_trans,
                                   int nr_oversampled_trans)
{
	int bid_y = blockIdx.y; //The image number id
	int index_in_row = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = bid_y * (nr_orients * nr_oversampled_rot * nr_trans * nr_oversampled_trans);
	if (index_in_row >= (nr_orients * nr_oversampled_rot * nr_trans * nr_oversampled_trans))
	{
		return ;
	}

	DOUBLE local_weight;

	int orients_index = (index_in_row) / (nr_oversampled_rot * nr_trans * nr_oversampled_trans);
	if (isSignificant_D[orients_index] == 1)
	{
		local_weight = exp_Mweight_D[bid_y * exp_Mweight_xdim + index_in_row];
		if (local_weight >= exp_Sum_Weigh_particles[bid_y])
		{
			valid_weight_list_D[offset + index_in_row] = 1;
		}
	}
}

void  sign_Weight_gpu(DOUBLE* exp_Mweight_D,
                      DOUBLE* exp_Sum_Weigh_particles,
                      int* isSignificant_D,
                      int* valid_weight_list_D,
                      int exp_Mweight_xdim,
                      int nr_images,
                      int nr_orients,
                      int nr_oversampled_rot,
                      int nr_trans,
                      int nr_oversampled_trans)
{
	int nr_blocks_x = (nr_orients * nr_oversampled_rot * nr_trans * nr_oversampled_trans + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(nr_blocks_x, nr_images, 1);
	sign_Weight_kernel <<< dimGrid, dimBlock>>>(exp_Mweight_D,
	                                            exp_Sum_Weigh_particles,
	                                            isSignificant_D,
	                                            valid_weight_list_D,
	                                            exp_Mweight_xdim,
	                                            nr_images,
	                                            nr_orients,
	                                            nr_oversampled_rot,
	                                            nr_trans,
	                                            nr_oversampled_trans);

}

__global__ void compact_Weight_kernel(DOUBLE* exp_Mweight_D,
                                      DOUBLE* exp_Sum_Weigh_particles,
                                      int* valid_weight_list_D,
                                      int* compact_position_list_D,
                                      int exp_Mweight_xdim,
                                      int nr_images,
                                      int nr_orients,
                                      int nr_oversampled_rot,
                                      int nr_trans,
                                      int nr_oversampled_trans,
                                      int iorientclass_offset,
                                      DOUBLE* valid_weight_D,
                                      int* weight_index,
                                      int* image_index_flag_D)
{
	int bid_x = blockIdx.x;
	int bid_y = blockIdx.y; //The image number id
	int index_in_row = threadIdx.x + bid_x * blockDim.x;
	int offset = bid_y * (nr_orients * nr_oversampled_rot * nr_trans * nr_oversampled_trans);
	if (index_in_row >= (nr_orients * nr_oversampled_rot * nr_trans * nr_oversampled_trans))
	{
		return ;
	}

	if (valid_weight_list_D[offset + index_in_row])
	{

		int out_position = compact_position_list_D[offset + index_in_row] ;
		valid_weight_D[out_position] = exp_Mweight_D[bid_y * exp_Mweight_xdim + index_in_row] / exp_Sum_Weigh_particles[bid_y];
		weight_index[out_position] =  offset + index_in_row;
		image_index_flag_D[out_position] =  bid_y;//( offset + index_in_row)/(nr_orients*nr_oversampled_rot*nr_trans*nr_oversampled_trans);
	}
}

void compact_Weight_gpu(DOUBLE* exp_Mweight_D,
                        DOUBLE* exp_Sum_Weigh_particles,
                        int* valid_weight_list_D,
                        int* compact_position_list_D,
                        int exp_Mweight_xdim,
                        int nr_images,
                        int nr_orients,
                        int nr_oversampled_rot,
                        int nr_trans,
                        int nr_oversampled_trans,
                        int iorientclass_offset,
                        DOUBLE* valid_weight_D,
                        int* weight_index,
                        int* image_index_flag_D)
{

	int nr_blocks_x = (nr_orients * nr_oversampled_rot * nr_trans * nr_oversampled_trans + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(nr_blocks_x, nr_images, 1);

	compact_Weight_kernel <<< dimGrid, dimBlock>>>(exp_Mweight_D,
	                                               exp_Sum_Weigh_particles,
	                                               valid_weight_list_D,
	                                               compact_position_list_D,
	                                               exp_Mweight_xdim,
	                                               nr_images,
	                                               nr_orients,
	                                               nr_oversampled_rot,
	                                               nr_trans,
	                                               nr_oversampled_trans,
	                                               iorientclass_offset,
	                                               valid_weight_D,
	                                               weight_index,
	                                               image_index_flag_D);
}

__global__ void re_initial_exp_weight_kernel(DOUBLE* exp_Mweight_D,
                                             bool* exp_Mcoarse_significant_D,
                                             DOUBLE* mini_weight_particel_D,
                                             int nr_images,
                                             int exp_Mweight_D_size,
                                             int exp_ipass)
{
	int bid_x = blockIdx.x;
	int part_id = blockIdx.y; //The image number id
	int index_in_row = threadIdx.x + bid_x * blockDim.x;
	if (index_in_row >= exp_Mweight_D_size)
	{
		return ;
	}
	DOUBLE mini_weight = mini_weight_particel_D[part_id];
	if (exp_Mweight_D[index_in_row + part_id * exp_Mweight_D_size] != mini_weight)
	{
		exp_Mweight_D[index_in_row + part_id * exp_Mweight_D_size] = 0.;
		if (exp_ipass == 0)
		{
			exp_Mcoarse_significant_D[index_in_row + part_id * exp_Mweight_D_size] = false;
		}
	}
	else
	{
		exp_Mweight_D[index_in_row + part_id * exp_Mweight_D_size] = 1.;
		if (exp_ipass == 0)
		{
			exp_Mcoarse_significant_D[index_in_row + part_id * exp_Mweight_D_size] = true;
		}
	}

}
void re_initial_exp_weight_gpu(DOUBLE* exp_Mweight_D,
                               bool* exp_Mcoarse_significant_D,
                               DOUBLE* mini_weight_particel_D,
                               int nr_images,
                               int exp_Mweight_D_size,
                               int exp_ipass)
{

	int nr_blocks_x = (exp_Mweight_D_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(nr_blocks_x, nr_images, 1);
	re_initial_exp_weight_kernel <<< dimGrid, dimBlock>>>(exp_Mweight_D,
	                                                      exp_Mcoarse_significant_D,
	                                                      mini_weight_particel_D,
	                                                      nr_images,
	                                                      exp_Mweight_D_size,
	                                                      exp_ipass);

}

__global__ void  calculate_minimal_weight_per_particle_kernel(const  DOUBLE* __restrict__ exp_Mweight_D,
                                                              DOUBLE* Mini_weight,
                                                              DOUBLE init_value,
                                                              int valid_size,
                                                              int exp_Mweight_D_size)
{
	__shared__ DOUBLE local_weight[1024];

	int tid = threadIdx.x;
	DOUBLE weight = 99e99; 
	int nr_loop;
	if (tid < valid_size)
	{
		if (exp_Mweight_D[tid + blockIdx.x * exp_Mweight_D_size] != init_value)
		{
			weight = exp_Mweight_D[tid + blockIdx.x * exp_Mweight_D_size];
		}
	}
	nr_loop  = (valid_size + blockDim.x - 1) / blockDim.x;
	for (int i = 1; i < nr_loop; i++)
	{
		if ((tid + i * blockDim.x) < valid_size)
		{
			if (exp_Mweight_D[tid + blockIdx.x * exp_Mweight_D_size + i * blockDim.x] < weight && exp_Mweight_D[tid + i * blockDim.x + blockIdx.x * exp_Mweight_D_size] != init_value)
			{
				weight = exp_Mweight_D[tid + i * blockDim.x + blockIdx.x * exp_Mweight_D_size];
			}
		}
	}

	local_weight [tid] =  weight;
	__syncthreads();

	for (int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
		if (tid < s && (local_weight[tid] > local_weight[tid + s]))
		{
			local_weight[tid] = local_weight[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();
	// write result for this block to global mem
	if (tid == 0)
	{
		Mini_weight[blockIdx.x] = local_weight[0];
	}

}
void calculate_minimal_weight_per_particle_gpu(DOUBLE* exp_Mweight_D,
                                               DOUBLE* Mini_weight,
                                               DOUBLE init_value,
                                               int valid_size,
                                               int nr_particles,
                                               int exp_Mweight_D_size)
{
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(nr_particles, 1, 1);
	calculate_minimal_weight_per_particle_kernel <<< dimGrid, dimBlock>>>(exp_Mweight_D,
	                                                                      Mini_weight,
	                                                                      init_value,
	                                                                      valid_size,
	                                                                      exp_Mweight_D_size);

}

__global__ void  calculate_sum_weight_per_particle_kernel(const  DOUBLE* __restrict__ exp_Mweight_D,
                                                          DOUBLE* sum_weight,
                                                          const  int* __restrict__ local_none_zero_number,
                                                          int* total_none_zero_number,
                                                          const  DOUBLE* __restrict__ max_weight_per_particle_D,
                                                          DOUBLE* max_weight_D,
                                                          int valid_size,
                                                          int exp_Mweight_D_size)
{
	__shared__ DOUBLE summer_weight[512];
	__shared__ int sum_none_zero[512];
	int tid = threadIdx.x;
	DOUBLE weight = 0.;
	int count = 0;
	int nr_loop;
	if (tid < valid_size)
	{
		count = local_none_zero_number[tid + blockIdx.x * exp_Mweight_D_size];
		weight = exp_Mweight_D[tid + blockIdx.x * exp_Mweight_D_size];
	}
	nr_loop  = (valid_size + blockDim.x - 1) / blockDim.x;
	for (int i = 1; i < nr_loop; i++)
	{
		if ((tid + i * blockDim.x) < valid_size)
		{
			weight += exp_Mweight_D[tid + i * blockDim.x + blockIdx.x * exp_Mweight_D_size];
			count += local_none_zero_number[tid + i * blockDim.x + blockIdx.x * exp_Mweight_D_size];
		}
	}

	summer_weight [tid] =  weight;
	sum_none_zero [tid] = count;
	__syncthreads();

	for (int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sum_none_zero[tid] += sum_none_zero[tid + s];
			summer_weight[tid] += summer_weight[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();
	// write result for this block to global mem
	if (tid == 0)
	{
		sum_weight[blockIdx.x] = summer_weight[0];
		total_none_zero_number[blockIdx.x] = sum_none_zero[0];
	}

}

void calculate_sum_weight_per_particle_gpu(DOUBLE* exp_Mweight_D,
                                           DOUBLE* sum_weight,
                                           int* local_none_zero_number,
                                           int* total_none_zero_number,
                                           DOUBLE* max_weight_per_particle_D,
                                           DOUBLE* max_weight_D,
                                           int valid_size,
                                           int nr_particles,
                                           int exp_Mweight_D_size)
{
	dim3 dimBlock(512, 1, 1);
	dim3 dimGrid(nr_particles, 1, 1);
	calculate_sum_weight_per_particle_kernel <<< dimGrid, dimBlock>>>(exp_Mweight_D,
	                                                                  sum_weight,
	                                                                  local_none_zero_number,
	                                                                  total_none_zero_number,
	                                                                  max_weight_per_particle_D,
	                                                                  max_weight_D,
	                                                                  valid_size,
	                                                                  exp_Mweight_D_size);
}


__global__ void calculate_significant_sum_kernel(DOUBLE* exp_Mweight_D,
                                                 DOUBLE* thresh_weight_D,
                                                 DOUBLE* exp_sum_weight_D,
                                                 int valid_size,
                                                 int exp_Mweight_D_size,
                                                 int down_factor
                                                )
{
	int bid_x = blockIdx.x;
	int ipart  = blockIdx.y;
	int tid = threadIdx.x;
	int index = bid_x * blockDim.x + tid;
	__shared__ DOUBLE local_sum_weight[BLOCK_SIZE];
	local_sum_weight[tid] = 0.;

	if (index >= valid_size)
	{
		return;
	}

	DOUBLE* thisthread_exp_Mweight_D = &exp_Mweight_D[ipart * exp_Mweight_D_size];

	if (thisthread_exp_Mweight_D[index] < (thresh_weight_D[ipart] / down_factor))
	{
		local_sum_weight[tid] = 0.;
	}
	else
	{
		local_sum_weight[tid] = thisthread_exp_Mweight_D[index];
	}

	//calculate the partial sum of each thread block
	__syncthreads();
	for (int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			local_sum_weight[tid] += local_sum_weight[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0)
	{
		exp_sum_weight_D[blockIdx.x + blockIdx.y * gridDim.x] = local_sum_weight[0];
	}

}


__global__ void  calculate_sum_significant_weight_per_particle_kernel(DOUBLE* exp_Mweight_D,
                                                                      DOUBLE* sum_weight,
                                                                      int valid_size,
                                                                      int exp_Mweight_D_size)
{
	__shared__ DOUBLE summer_weight[512];
	int tid = threadIdx.x;
	DOUBLE weight = 0.;
	int nr_loop;
	if (tid < valid_size)
	{
		weight = exp_Mweight_D[tid + blockIdx.x * exp_Mweight_D_size];
	}
	nr_loop  = (valid_size + blockDim.x - 1) / blockDim.x;
	for (int i = 1; i < nr_loop; i++)
	{
		if ((tid + i * blockDim.x) < valid_size)
		{
			weight += exp_Mweight_D[tid + i * blockDim.x + blockIdx.x * exp_Mweight_D_size];
		}
	}

	summer_weight [tid] =  weight;
	__syncthreads();

	for (int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			summer_weight[tid] += summer_weight[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0)
	{
		sum_weight[blockIdx.x] = summer_weight[0];
	}

}

void calculate_significant_partial_sum_gpu(DOUBLE* exp_Mweight_D,
                                           DOUBLE* thresh_weight_D,
                                           DOUBLE* sum_weight,
                                           int valid_size,
                                           int nr_particles,
                                           int exp_Mweight_D_size,
                                           int down_factor)
{
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((valid_size + BLOCK_SIZE - 1) / BLOCK_SIZE , nr_particles, 1);
	calculate_significant_sum_kernel <<< dimGrid, dimBlock>>> (exp_Mweight_D,
	                                                           thresh_weight_D,
	                                                           sum_weight,
	                                                           valid_size,
	                                                           exp_Mweight_D_size, down_factor);

}
void calculate_sum_significant_weight_per_particle_gpu(DOUBLE* exp_Mweight_D,
                                                       DOUBLE* sum_weight,
                                                       int valid_size,
                                                       int nr_particles,
                                                       int exp_Mweight_D_size)
{
	dim3 dimBlock(512, 1, 1);
	dim3 dimGrid(nr_particles, 1, 1);
	calculate_sum_significant_weight_per_particle_kernel <<< dimGrid, dimBlock>>>(exp_Mweight_D,
	                                                                              sum_weight,
	                                                                              valid_size,
	                                                                              exp_Mweight_D_size);
}
extern __shared__ DOUBLE  diff_array[ ];

__global__ void calculate_diff2_no_do_squared_difference_pass0_kernel(DOUBLE* diff2_D,
                                                                      const CUFFT_COMPLEX * __restrict__ frefctf_D,
                                                                      const CUFFT_COMPLEX * __restrict__ Fimg_shift_D,
                                                                      const  DOUBLE* __restrict__  exp_local_sqrtXi2_D,
                                                                      const  int* __restrict__  valid_orient_trans_index_D,
                                                                      int exp_nr_particles,
                                                                      int exp_nr_orients,
                                                                      int exp_nr_trans,
                                                                      int exp_nr_oversampled_rot,
                                                                      int exp_nr_oversampled_trans,
                                                                      int nr_valid_orients,
                                                                      int diff_xdim,
                                                                      int image_size)
{
	int bid_x = blockIdx.x;
	int part_id = blockIdx.y;
	int tid = threadIdx.x;
	int orient_id = (valid_orient_trans_index_D[bid_x]) / (exp_nr_trans);
	int trans_id = (valid_orient_trans_index_D[bid_x] % exp_nr_trans);

	__shared__ DOUBLE diff2_array[BLOCK_SIZE], suma2_array[BLOCK_SIZE];
	diff2_array[tid] = 0.;
	suma2_array[tid] = 0.;

	DOUBLE suma2_real = 0., suma2_imag = 0.;


	int shift_id = part_id *  exp_nr_trans + trans_id;
	int reordered_orient_id = bid_x / exp_nr_trans;//valid_orient_prefix_sum[orient_id];
	int ref_image_offset = (part_id * nr_valid_orients + reordered_orient_id) * image_size;

	const  CUFFT_COMPLEX * __restrict__ thisthread_Frefctf_D = &frefctf_D[ref_image_offset]; //(part_id+(reordered_orient_id*exp_nr_oversampled_rot+oversampled_rot_id)*exp_nr_particles)
	const  CUFFT_COMPLEX * __restrict__ thisthread_Fimg_shift_D = &Fimg_shift_D[shift_id * image_size];  //shift_id * image_size

	for (int i = tid; i < image_size; i += BLOCK_SIZE)
	{

		diff2_array[tid] += (thisthread_Frefctf_D[i].x * thisthread_Fimg_shift_D[i].x + thisthread_Frefctf_D[i].y * thisthread_Fimg_shift_D[i].y);

		suma2_real = thisthread_Frefctf_D[i].x;
		suma2_imag = thisthread_Frefctf_D[i].y;
		suma2_array[tid] += suma2_real * suma2_real + suma2_imag * suma2_imag;
	}
	__syncthreads();
	for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			diff2_array[tid] += diff2_array[tid + s];
			suma2_array[tid] += suma2_array[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		diff2_D[part_id * diff_xdim + orient_id * exp_nr_trans + trans_id] = - diff2_array[0] / (sqrt(suma2_array[0]) * exp_local_sqrtXi2_D[part_id]);
	}
}
void calculate_diff2_no_do_squared_difference_pass0_gpu(DOUBLE* diff2_D,
                                                        CUFFT_COMPLEX * frefctf_D,
                                                        CUFFT_COMPLEX * Fimg_shift_D,
                                                        DOUBLE* exp_local_sqrtXi2_D,
                                                        int* valid_orient_trans_index_D,
                                                        int exp_nr_particles,
                                                        int exp_nr_orients,
                                                        int exp_nr_trans,
                                                        int exp_nr_oversampled_rot,
                                                        int exp_nr_oversampled_trans,
                                                        int nr_valid_orients,
                                                        int diff_xdim,
                                                        int image_size)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(nr_valid_orients * exp_nr_trans, exp_nr_particles, 1);

	calculate_diff2_no_do_squared_difference_pass0_kernel <<< dimGrid, dimBlock>>>(diff2_D,
	                                                                               frefctf_D,
	                                                                               Fimg_shift_D,
	                                                                               exp_local_sqrtXi2_D,
	                                                                               valid_orient_trans_index_D,
	                                                                               exp_nr_particles,
	                                                                               exp_nr_orients,
	                                                                               exp_nr_trans,
	                                                                               exp_nr_oversampled_rot,
	                                                                               exp_nr_oversampled_trans,
	                                                                               nr_valid_orients,
	                                                                               diff_xdim,
	                                                                               image_size);
}



__global__ void calculate_diff2_do_squared_difference_pass0_kernel(DOUBLE* diff2_D,
                                                                   const  CUFFT_COMPLEX * __restrict__  frefctf_D,
                                                                   const  CUFFT_COMPLEX * __restrict__  Fimg_shift_D,
                                                                   const  DOUBLE* __restrict__  exp_highres_Xi2_imgs_D,
                                                                   const  DOUBLE* __restrict__  Minvsigma2_D,
                                                                   const  int* __restrict__ valid_orient_trans_index_D,
                                                                   int exp_nr_particles,
                                                                   int exp_nr_orients,
                                                                   int exp_nr_trans,
                                                                   int exp_nr_oversampled_rot,
                                                                   int exp_nr_oversampled_trans,
                                                                   int nr_valid_orients,
                                                                   int diff_xdim,
                                                                   int image_size,
                                                                   int nr_trans_loop,
                                                                   int nr_trans_stride)

{
	DOUBLE* diff2_array = (DOUBLE*) &diff_array;

	int part_id = blockIdx.y;
	int bid_x = blockIdx.x;
	int tid = threadIdx.x;

	int orient_id = (valid_orient_trans_index_D[bid_x * (exp_nr_trans * exp_nr_particles) + part_id * exp_nr_trans] % (exp_nr_orients * exp_nr_trans)) / (exp_nr_trans);
	int ref_image_offset = (part_id * nr_valid_orients + bid_x) * image_size;

	const  CUFFT_COMPLEX * __restrict__ thisthread_Frefctf_D = &frefctf_D[ref_image_offset];
	const  DOUBLE* __restrict__ thisthread_Minvsigma2_D = &Minvsigma2_D[part_id * image_size];
	DOUBLE refctf_real, refctf_img , minvsigma2;
	DOUBLE Xi2_imgs = exp_highres_Xi2_imgs_D[part_id];
	if (nr_trans_loop == 1)
	{
		DOUBLE diff2_real = 0., diff2_imag = 0.;
		int exp_trans_offset = part_id * exp_nr_trans ;
		for (int trans_id = 0; trans_id < exp_nr_trans; trans_id++)
		{
			diff2_array[tid + trans_id * blockDim.x ] = 0.;
		}

		for (int i = tid; i < image_size; i += blockDim.x)
		{
			refctf_real = thisthread_Frefctf_D[i].x;
			refctf_img = thisthread_Frefctf_D[i].y;
			minvsigma2 = thisthread_Minvsigma2_D[i];
#pragma unroll
			for (int trans_id = 0; trans_id < exp_nr_trans; trans_id++)
			{
				diff2_real = refctf_real - Fimg_shift_D[(exp_trans_offset + trans_id) * image_size + i].x;
				diff2_imag = refctf_img - Fimg_shift_D[(exp_trans_offset + trans_id) * image_size + i].y;
				diff2_array[tid + trans_id * blockDim.x ] += (diff2_real * diff2_real + diff2_imag * diff2_imag) * 0.5 * minvsigma2;
			}
		}
		__syncthreads();

		for (int s = blockDim.x >> 1; s > 0; s >>= 1)
		{
			if (tid < s)
			{
#pragma unroll
				for (int trans_id = 0; trans_id < exp_nr_trans; trans_id++)
				{
					diff2_array[tid + trans_id * blockDim.x] += diff2_array[tid + trans_id * blockDim.x + s];
				}
			}
			__syncthreads();
		}

		if (tid < exp_nr_trans)
		{
			diff2_D[part_id * diff_xdim + orient_id * exp_nr_trans + tid] = diff2_array[tid * blockDim.x] + Xi2_imgs * 0.5;
		}
	}
	else
	{
		for (int j = 0; j < nr_trans_loop; j++)
		{
			int nr_trans = (j < (nr_trans_loop - 1)) ? nr_trans_stride : (exp_nr_trans - nr_trans_stride * j);
			DOUBLE diff2_real = 0., diff2_imag = 0.;
			int exp_trans_offset = part_id * exp_nr_trans + nr_trans_stride * j;
			for (int trans_id = 0; trans_id < nr_trans; trans_id++)
			{
				diff2_array[tid + trans_id * blockDim.x ] = 0.;
			}
			for (int i = tid; i < image_size; i += blockDim.x)
			{
				refctf_real = thisthread_Frefctf_D[i].x;
				refctf_img = thisthread_Frefctf_D[i].y;
				minvsigma2 = thisthread_Minvsigma2_D[i];
#pragma unroll
				for (int trans_id = 0; trans_id < nr_trans; trans_id++)
				{
					diff2_real = refctf_real - Fimg_shift_D[(exp_trans_offset + trans_id) * image_size + i].x;
					diff2_imag = refctf_img - Fimg_shift_D[(exp_trans_offset + trans_id) * image_size + i].y;
					diff2_array[tid + trans_id * blockDim.x ] += (diff2_real * diff2_real + diff2_imag * diff2_imag) * 0.5 * minvsigma2;
				}
			}
			__syncthreads();

			for (int s = blockDim.x >> 1; s > 0; s >>= 1)
			{
				if (tid < s)
				{
					for (int trans_id = 0; trans_id < nr_trans; trans_id++)
					{
						diff2_array[tid + trans_id * blockDim.x] += diff2_array[tid + trans_id * blockDim.x + s];
					}
				}
				__syncthreads();
			}
			if (tid < nr_trans)
			{
				diff2_D[part_id * diff_xdim + orient_id * exp_nr_trans + tid + nr_trans_stride * j] = diff2_array[tid * blockDim.x] + Xi2_imgs * 0.5;
			}
		}
	}

}

void calculate_diff2_do_squared_difference_pass0_gpu(DOUBLE* diff2_D,
                                                     CUFFT_COMPLEX * frefctf_D,
                                                     CUFFT_COMPLEX * Fimg_shift_D,
                                                     DOUBLE* exp_highres_Xi2_imgs_D,
                                                     DOUBLE* Minvsigma2_D,
                                                     int* valid_orient_trans_index_D,
                                                     int exp_nr_particles,
                                                     int exp_nr_orients,
                                                     int exp_nr_trans,
                                                     int exp_nr_oversampled_rot,
                                                     int exp_nr_oversampled_trans,
                                                     int nr_valid_orients,
                                                     int diff_xdim,
                                                     int image_size)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(nr_valid_orients, exp_nr_particles, 1);
	int shared_size;
	int nr_trans_stride = 4;
	if (exp_nr_trans <= 4)
	{
		shared_size = sizeof(DOUBLE) * (BLOCK_SIZE_128 * exp_nr_trans);
	}
	else
	{
		shared_size = sizeof(DOUBLE) * (BLOCK_SIZE_128 * nr_trans_stride);
	}
	int nr_trans_loop = (exp_nr_trans <= 4) ? 1 : ((exp_nr_trans + (nr_trans_stride - 1)) / nr_trans_stride);
	calculate_diff2_do_squared_difference_pass0_kernel <<< dimGrid, dimBlock, shared_size>>>(diff2_D,
	        frefctf_D,
	        Fimg_shift_D,
	        exp_highres_Xi2_imgs_D,
	        Minvsigma2_D,
	        valid_orient_trans_index_D,
	        exp_nr_particles,
	        exp_nr_orients,
	        exp_nr_trans,
	        exp_nr_oversampled_rot,
	        exp_nr_oversampled_trans,
	        nr_valid_orients,
	        diff_xdim,
	        image_size,
	        nr_trans_loop,
	        nr_trans_stride);





}
__global__ void apply_ctf_and_calculate_all_diff2_squared_pass0_kernel(const  CUFFT_COMPLEX * __restrict__ fref_D,
                                                                       const  DOUBLE* __restrict__ mctf_D,
                                                                       const  DOUBLE* __restrict__ myscale_D,
                                                                       DOUBLE* diff2_D,
                                                                       //const  CUFFT_COMPLEX  * __restrict__  frefctf_D,
                                                                       const  CUFFT_COMPLEX * __restrict__  Fimg_shift_D,
                                                                       const  DOUBLE* __restrict__  exp_highres_Xi2_imgs_D,
                                                                       const  DOUBLE* __restrict__  Minvsigma2_D,
                                                                       const  int* __restrict__ valid_orient_trans_index_D,
                                                                       int exp_nr_particles,
                                                                       int exp_nr_orients,
                                                                       int exp_nr_trans,
                                                                       int exp_nr_oversampled_rot,
                                                                       int exp_nr_oversampled_trans,
                                                                       int nr_valid_orients,
                                                                       int diff_xdim,
                                                                       int image_size,
                                                                       bool do_ctf_correction_and_refs_are_ctf_corrected,
                                                                       bool do_scale_correction)
{
	DOUBLE* diff2_array = (DOUBLE*) &diff_array;

	int part_id = blockIdx.y;
	int tid = threadIdx.x;
	int bid_x = blockIdx.x;
	int orient_id = (valid_orient_trans_index_D[bid_x] % (exp_nr_orients * exp_nr_trans)) / (exp_nr_trans);

	int offset_image = part_id * image_size ;
	int offset_orientation = blockIdx.x * image_size;

	DOUBLE refctf_real, refctf_img , minvsigma2;
	DOUBLE diff2_real , diff2_imag;

	for (int trans_id = 0; trans_id < exp_nr_trans; trans_id++)
	{
		diff2_array[tid + trans_id * blockDim.x ] = 0.;
	}

	for (int i = tid; i < image_size; i += blockDim.x)
	{
		refctf_real = fref_D[i + offset_orientation].x;
		refctf_img = fref_D[i + offset_orientation].y;

		if (do_ctf_correction_and_refs_are_ctf_corrected)
		{
			refctf_real *= mctf_D[i + offset_image];
			refctf_img *= mctf_D[i + offset_image];
		}
		if (do_scale_correction)
		{
			refctf_real *= myscale_D[part_id];
			refctf_img *= myscale_D[part_id];
		}
		minvsigma2 = Minvsigma2_D[part_id * image_size + i];
		for (int trans_id = 0; trans_id < exp_nr_trans; trans_id++)
		{
			int shift_id = part_id * exp_nr_trans + trans_id;
			Fimg_shift_D[shift_id * image_size];
			diff2_real = refctf_real - Fimg_shift_D[shift_id * image_size + i].x;
			diff2_imag = refctf_img - Fimg_shift_D[shift_id * image_size + i].y;
			diff2_array[tid + trans_id * blockDim.x ] += (diff2_real * diff2_real + diff2_imag * diff2_imag) * 0.5 * minvsigma2;
		}
	}
	__syncthreads();

	for (int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
#pragma unroll
			for (int trans_id = 0; trans_id < exp_nr_trans; trans_id++)
			{
				diff2_array[tid + trans_id * blockDim.x] += diff2_array[tid + trans_id * blockDim.x + s];
			}
		}
		__syncthreads();
	}
	__syncthreads();

	if (tid < exp_nr_trans)
	{
		diff2_D[part_id * diff_xdim + orient_id * exp_nr_trans + tid] = diff2_array[tid * blockDim.x] + exp_highres_Xi2_imgs_D[part_id] * 0.5;
	}

}
void apply_ctf_and_calculate_all_diff2_squared_pass0_gpu(CUFFT_COMPLEX * fref_D,
                                                         DOUBLE* exp_local_Fctfs_D,
                                                         DOUBLE* myscale_D,
                                                         DOUBLE* diff2_D,
                                                         CUFFT_COMPLEX * Fimg_shift_D,
                                                         DOUBLE* exp_highres_Xi2_imgs_D,
                                                         DOUBLE* Minvsigma2_D,
                                                         int* valid_orient_trans_index_D,
                                                         int exp_nr_particles,
                                                         int exp_nr_orients,
                                                         int exp_nr_trans,
                                                         int exp_nr_oversampled_rot,
                                                         int exp_nr_oversampled_trans,
                                                         int nr_valid_orients,
                                                         int diff_xdim,
                                                         int image_size,
                                                         bool do_ctf_correction_and_refs_are_ctf_corrected,
                                                         bool do_scale_correction)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(nr_valid_orients, exp_nr_particles, 1);
	int shared_size = sizeof(DOUBLE) * (BLOCK_SIZE_128 * exp_nr_trans);
	apply_ctf_and_calculate_all_diff2_squared_pass0_kernel <<< dimGrid, dimBlock, shared_size>>>
	(fref_D,
	 exp_local_Fctfs_D,
	 myscale_D,
	 diff2_D,
	 Fimg_shift_D,
	 exp_highres_Xi2_imgs_D,
	 Minvsigma2_D,
	 valid_orient_trans_index_D,
	 exp_nr_particles,
	 exp_nr_orients,
	 exp_nr_trans,
	 exp_nr_oversampled_rot,
	 exp_nr_oversampled_trans,
	 nr_valid_orients,
	 diff_xdim,
	 image_size,
	 do_ctf_correction_and_refs_are_ctf_corrected,
	 do_scale_correction);

}

__global__ void calculate_img_power_kernel(const  CUFFT_COMPLEX * __restrict__ local_Faux_D,
                                           DOUBLE* exp_power_imgs_D,
                                           DOUBLE* exp_highres_Xi2_imgs_D,
                                           int im_x,
                                           int im_y,
                                           int im_z,
                                           int im_xy,
                                           int im_size,
                                           int half_mymodel_ori_size,
                                           int half_mymodel_current_size
                                          )
{
	int tid = threadIdx.x;
	int pixel_id = tid + blockIdx.x * blockDim.x;
	int im_id = blockIdx.y;

	__shared__ DOUBLE normFaux_local[BLOCK_SIZE_128];
	normFaux_local[tid] = 0.;
	if (pixel_id >= im_size)
	{
		return;
	}

	int jp = pixel_id % im_x;
	int ip = (pixel_id / im_x) % im_y;
	int kp = pixel_id / (im_xy);

	ip  = (ip < im_x) ? (ip) : (ip - im_y);
	kp = (kp < im_x) ? (kp) : (kp - im_z);

	DOUBLE ires = sqrt((DOUBLE)(kp * kp + ip * ip + jp * jp));
	int ires_id = (((ires) > 0) ? (int)((ires) + 0.5) : (int)((ires) - 0.5));


	if (ires_id > 0 && ires_id < half_mymodel_ori_size && !(jp == 0 && ip < 0))
	{
		DOUBLE normFaux  = (local_Faux_D[pixel_id + im_id * im_size].x * local_Faux_D[pixel_id + im_id * im_size].x)
		                   + (local_Faux_D[pixel_id + im_id * im_size].y * local_Faux_D[pixel_id + im_id * im_size].y);
		atomicAdd(&(exp_power_imgs_D[im_id * half_mymodel_ori_size + ires_id]),  normFaux);
		if (ires_id >= half_mymodel_current_size)
		{
			normFaux_local[tid] = normFaux;
		}
		//c

	}
	__syncthreads();
	for (int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			normFaux_local[tid] += normFaux_local[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		atomicAdd(&(exp_highres_Xi2_imgs_D[im_id]),  normFaux_local[0]);
	}
}

void calculate_img_power_gpu(CUFFT_COMPLEX * local_Faux_D,
                             DOUBLE* exp_power_imgs_D,
                             DOUBLE* exp_highres_Xi2_imgs_D,
                             int nr_images,
                             int Faux_x,
                             int Faux_y,
                             int Faux_z,
                             int mymodel_ori_size,
                             int mymodel_current_size
                            )
{
	if (mymodel_current_size < mymodel_ori_size)
	{
		int image_size = Faux_z * Faux_y * Faux_x;
		int blk_x = (image_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128;
		dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
		dim3 dimGrid(blk_x, nr_images, 1);
		calculate_img_power_kernel <<< dimGrid, dimBlock>>>(local_Faux_D,
		                                                    exp_power_imgs_D,
		                                                    exp_highres_Xi2_imgs_D,
		                                                    Faux_x,
		                                                    Faux_y,
		                                                    Faux_z,
		                                                    Faux_x * Faux_y,
		                                                    Faux_x * Faux_y * Faux_z,
		                                                    mymodel_ori_size / 2 + 1,
		                                                    mymodel_current_size / 2 + 1);

	}
	else
	{
		cudaMemset(exp_highres_Xi2_imgs_D, 0., sizeof(DOUBLE) *nr_images);
	}

}

__device__ DOUBLE getCTF_kernel(int im, DOUBLE X, DOUBLE Y,
                                bool do_abs = false, bool do_only_flip_phases = false, bool do_intact_until_first_peak = false, bool do_damping = true)

{
	DOUBLE u2 = X * X + Y * Y;
	DOUBLE u = sqrt(u2);
	DOUBLE u4 = u2 * u2;
	DOUBLE deltaf;
	if (ABS(X) < XMIPP_EQUAL_ACCURACY &&
	        ABS(Y) < XMIPP_EQUAL_ACCURACY)
	{
		deltaf = 0.;
	}
	else
	{
		DOUBLE ellipsoid_ang = atan2(Y, X) -  ctf_related_parameters_D[im * 9 + 2];
		DOUBLE cos_ellipsoid_ang_2 = cos(2 * ellipsoid_ang);
		deltaf = (ctf_related_parameters_D[im * 9] +  ctf_related_parameters_D[im * 9 + 1] * cos_ellipsoid_ang_2);
	}
	DOUBLE argument = ctf_related_parameters_D[im * 9 + 5] * deltaf * u2 + ctf_related_parameters_D[im * 9 + 6]  * u4;
	DOUBLE retval;
	if (do_intact_until_first_peak && ABS(argument) < PI / 2.)
	{
		retval = 1.;
	}
	else
	{
		retval = -(ctf_related_parameters_D[im * 9 + 7] * sin(argument) - ctf_related_parameters_D[im * 9 + 3] * cos(argument)); // Q0 should be positive
	}
	if (do_damping)
	{
		DOUBLE E = exp(ctf_related_parameters_D[im * 9 + 8]  * u2); // B-factor decay (K4 = -Bfac/4);
		retval *= E;
	}
	if (do_abs)
	{
		retval = ABS(retval);
	}
	else if (do_only_flip_phases)
	{
		retval = (retval < 0.) ? -1. : 1.;
	}
	return ctf_related_parameters_D[im * 9 + 4]  * retval;
}

__global__ void getFftwImage_kernel(DOUBLE* local_Fctf_images_D,
                                    DOUBLE orixdim_angpix, DOUBLE oriydim_angpix,
                                    bool do_abs, bool do_only_flip_phases, bool do_intact_until_first_peak,
                                    bool do_damping,
                                    int nr_images,
                                    int xdim,
                                    int ydim,
                                    bool do_ctf_correction)
{
	int tid = threadIdx.x;
	int pixel_id = tid + blockIdx.x * blockDim.x;
	int im_id = blockIdx.y;

	if (pixel_id >= xdim * ydim)
	{
		return;
	}
	if (do_ctf_correction)
	{
		int jp = pixel_id % xdim;
		int ip = (pixel_id / xdim);

		ip  = (ip < xdim) ? (ip) : (ip - ydim);
		DOUBLE x = (DOUBLE)jp / orixdim_angpix;
		DOUBLE y = (DOUBLE)ip / oriydim_angpix;
		local_Fctf_images_D[im_id * xdim * ydim + pixel_id] = getCTF_kernel(im_id, x, y, do_abs, do_only_flip_phases, do_intact_until_first_peak, do_damping);
	}
	else
	{
		local_Fctf_images_D[im_id * xdim * ydim + pixel_id] = 1.0;
	}

}

void getFftwImage_gpu(DOUBLE* local_Fctf_images_D,
                      DOUBLE* ctf_related_parameters_H,
                      int orixdim, int oriydim, DOUBLE angpix,
                      bool do_abs, bool do_only_flip_phases, bool do_intact_until_first_peak,
                      bool do_damping,
                      int nr_images,
                      int xdim,
                      int ydim,
                      bool do_ctf_correction)
{
	DOUBLE orixdim_angpix = (DOUBLE) orixdim * angpix;
	DOUBLE oriydim_angpix = (DOUBLE) oriydim * angpix;
	int image_size = xdim * ydim;
	int blk_x = (image_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128;
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(blk_x, nr_images, 1);
	cudaMemcpyToSymbol(ctf_related_parameters_D, ctf_related_parameters_H, nr_images * 9 * sizeof(DOUBLE), 0 , cudaMemcpyHostToDevice);
	getFftwImage_kernel <<< dimGrid, dimBlock>>>(local_Fctf_images_D,
	                                             orixdim_angpix,
	                                             oriydim_angpix,
	                                             do_abs,
	                                             do_only_flip_phases,
	                                             do_intact_until_first_peak,
	                                             do_damping,
	                                             nr_images,
	                                             xdim,
	                                             ydim,
	                                             do_ctf_correction);



}
__global__ void ScaleComplexPointwise_kernel(CUFFT_COMPLEX * a, int size, DOUBLE scale)
{
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_index >= size)
	{
		return;
	}
	a[global_index].x *= scale;
	a[global_index].y *= scale;
}

void ScaleComplexPointwise_gpu(CUFFT_COMPLEX * fFourier_D, int size, DOUBLE scale)
{
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	ScaleComplexPointwise_kernel <<< dimGrid, blockDim>>>(fFourier_D, size,  scale);
}



__global__ void calculate_diff2_no_do_squared_difference_mask_kernel(DOUBLE* diff2_D,
                                                                CUFFT_COMPLEX * frefctf_yellow_D, CUFFT_COMPLEX * frefctf_red_D,
                                                                CUFFT_COMPLEX * Fimg_shift_D,
                                                                DOUBLE* exp_local_sqrtXi2_D,
                                                                int* valid_orient_trans_index_D,
                                                                int* valid_orient_prefix_sum,
                                                                int exp_nr_particles,
                                                                int exp_nr_orients,
                                                                int exp_nr_trans,
                                                                int exp_nr_oversampled_rot,
                                                                int exp_nr_oversampled_trans,
                                                                int nr_valid_orients,
                                                                int diff_xdim,
                                                                int image_size)
{
	int bid_x = blockIdx.x;
	int part_id = valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] / (exp_nr_orients * exp_nr_trans);
	int tid = threadIdx.x;
	int orient_id = (valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] % (exp_nr_orients * exp_nr_trans)) / (exp_nr_trans);
	int trans_id = (valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] % exp_nr_trans);

	__shared__ DOUBLE diff2_array[BLOCK_SIZE], suma2_array[BLOCK_SIZE];
	diff2_array[tid] = 0.;
	suma2_array[tid] = 0.;

	DOUBLE suma2_real = 0., suma2_imag = 0.;

	int oversampled_rot_id =  bid_x % (exp_nr_oversampled_rot * exp_nr_oversampled_trans) / exp_nr_oversampled_trans;
	int oversampled_trans_id =  bid_x % exp_nr_oversampled_trans;
	int shift_id = part_id * exp_nr_oversampled_trans * exp_nr_trans +
	               trans_id * exp_nr_oversampled_trans + oversampled_trans_id;
	int reordered_orient_id = valid_orient_prefix_sum[orient_id];
	int ref_image_offset = (part_id * nr_valid_orients * exp_nr_oversampled_rot + reordered_orient_id * exp_nr_oversampled_rot + oversampled_rot_id) * image_size;

	const  CUFFT_COMPLEX * __restrict__ thisthread_Frefctf_yellow_D = &frefctf_yellow_D[ref_image_offset]; //(part_id+(reordered_orient_id*exp_nr_oversampled_rot+oversampled_rot_id)*exp_nr_particles)
	const  CUFFT_COMPLEX * __restrict__ thisthread_Frefctf_red_D = &frefctf_red_D[ref_image_offset]; //(part_id+(reordered_orient_id*exp_nr_oversampled_rot+oversampled_rot_id)*exp_nr_particles)
	const  CUFFT_COMPLEX * __restrict__ thisthread_Fimg_shift_D = &Fimg_shift_D[shift_id * image_size];  //shift_id * image_size


	for (int i = tid; i < image_size; i += BLOCK_SIZE)
	{

		diff2_array[tid] += thisthread_Frefctf_red_D[i].x * (thisthread_Fimg_shift_D[i].x - thisthread_Frefctf_yellow_D[i].x) 
			+ thisthread_Frefctf_red_D[i].y * (thisthread_Fimg_shift_D[i].y - thisthread_Frefctf_yellow_D[i].y);

		suma2_real = thisthread_Frefctf_red_D[i].x;
		suma2_imag = thisthread_Frefctf_red_D[i].y;
		suma2_array[tid] += suma2_real * suma2_real + suma2_imag * suma2_imag;
	}
	__syncthreads();
	for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			diff2_array[tid] += diff2_array[tid + s];
			suma2_array[tid] += suma2_array[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();

	if (tid == 0)
	{
		diff2_D[part_id * diff_xdim + ((orient_id * exp_nr_trans + trans_id)* exp_nr_oversampled_rot + oversampled_rot_id)*  exp_nr_oversampled_trans + oversampled_trans_id] = - diff2_array[0] / (sqrt(suma2_array[0]) * exp_local_sqrtXi2_D[part_id]);
	}
}

void calculate_diff2_no_do_squared_difference_mask_gpu(DOUBLE* diff2_D,
                                                  CUFFT_COMPLEX * frefctf_yellow_D, CUFFT_COMPLEX *frefctf_red_D,
                                                  CUFFT_COMPLEX * Fimg_shift_D,
                                                  DOUBLE* exp_local_sqrtXi2_D,
                                                  int* valid_orient_trans_index_D,
                                                  int* valid_orient_prefix_sum,
                                                  int exp_nr_particles,
                                                  int exp_nr_orients,
                                                  int exp_nr_trans,
                                                  int exp_nr_oversampled_rot,
                                                  int exp_nr_oversampled_trans,
                                                  int nr_valid_orients,
                                                  int diff_xdim,
                                                  int image_size,
                                                  int nr_valid_orient_trans)
{
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(exp_nr_oversampled_rot * exp_nr_oversampled_trans * nr_valid_orient_trans, 1, 1);

	calculate_diff2_no_do_squared_difference_mask_kernel <<< dimGrid, dimBlock>>>(diff2_D,
	                                                                         frefctf_yellow_D, frefctf_red_D,
	                                                                         Fimg_shift_D,
	                                                                         exp_local_sqrtXi2_D,
	                                                                         valid_orient_trans_index_D,
	                                                                         valid_orient_prefix_sum,
	                                                                         exp_nr_particles,
	                                                                         exp_nr_orients,
	                                                                         exp_nr_trans,
	                                                                         exp_nr_oversampled_rot,
	                                                                         exp_nr_oversampled_trans,
	                                                                         nr_valid_orients,
	                                                                         diff_xdim,
	                                                                         image_size);
}



__global__ void calculate_diff2_do_squared_difference_mask_kernel(DOUBLE* diff2_D,
                                                             const  CUFFT_COMPLEX * __restrict__  frefctf_yellow_D, 
                                                             const  CUFFT_COMPLEX * __restrict__  frefctf_red_D,
                                                             const  CUFFT_COMPLEX * __restrict__  Fimg_shift_D,
                                                             DOUBLE* exp_highres_Xi2_imgs_D,
                                                             DOUBLE* Minvsigma2_D,
                                                             int* valid_orient_trans_index_D,
                                                             int* valid_orient_prefix_sum,
                                                             int exp_nr_particles,
                                                             int exp_nr_orients,
                                                             int exp_nr_trans,
                                                             int exp_nr_oversampled_rot,
                                                             int exp_nr_oversampled_trans,
                                                             int nr_valid_orients,
                                                             int diff_xdim,
                                                             int image_size)
{

	int bid_x = blockIdx.x;
	int part_id = valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] / (exp_nr_orients * exp_nr_trans);
	int tid = threadIdx.x;
	int orient_id = (valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] % (exp_nr_orients * exp_nr_trans)) / (exp_nr_trans);
	int trans_id = (valid_orient_trans_index_D[bid_x / (exp_nr_oversampled_rot * exp_nr_oversampled_trans)] % exp_nr_trans);

	int oversampled_rot_id =  bid_x % (exp_nr_oversampled_rot * exp_nr_oversampled_trans) / exp_nr_oversampled_trans;
	int oversampled_trans_id =  bid_x % exp_nr_oversampled_trans;
	int shift_id = part_id * exp_nr_oversampled_trans * exp_nr_trans + trans_id * exp_nr_oversampled_trans + oversampled_trans_id;

	int reordered_orient_id = valid_orient_prefix_sum[orient_id];
	int ref_image_offset = (part_id * nr_valid_orients * exp_nr_oversampled_rot + reordered_orient_id * exp_nr_oversampled_rot + oversampled_rot_id) * image_size;

	__shared__ DOUBLE diff2_array[BLOCK_SIZE_128];
	diff2_array[tid] = 0.;

	const  CUFFT_COMPLEX * __restrict__ thisthread_Frefctf_yellow_D = &frefctf_yellow_D[ref_image_offset];
	const  CUFFT_COMPLEX * __restrict__ thisthread_Frefctf_red_D = &frefctf_red_D[ref_image_offset];
	const  CUFFT_COMPLEX * __restrict__ thisthread_Fimg_shift_D = &Fimg_shift_D[shift_id * image_size];

	DOUBLE* thisthread_Minvsigma2_D = &Minvsigma2_D[part_id * image_size];

	DOUBLE diff2_real = 0., diff2_imag = 0.;
	for (int i = tid; i < image_size; i += blockDim.x)
	{
		diff2_real = thisthread_Frefctf_yellow_D[i].x + thisthread_Frefctf_red_D[i].x - thisthread_Fimg_shift_D[i].x;
		diff2_imag = thisthread_Frefctf_yellow_D[i].y + thisthread_Frefctf_red_D[i].y - thisthread_Fimg_shift_D[i].y;
		diff2_array[tid] += (diff2_real * diff2_real + diff2_imag * diff2_imag) * 0.5 * thisthread_Minvsigma2_D[i];
	}

	__syncthreads();
	for (int s = BLOCK_SIZE_128 >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			diff2_array[tid] += diff2_array[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		diff2_D[part_id * diff_xdim + ((orient_id * exp_nr_trans + trans_id)* exp_nr_oversampled_rot + oversampled_rot_id)*  exp_nr_oversampled_trans + oversampled_trans_id] = diff2_array[0] + exp_highres_Xi2_imgs_D[part_id] * 0.5;
	}
}


void calculate_diff2_do_squared_difference_mask_gpu(DOUBLE* diff2_D,
                                               CUFFT_COMPLEX * frefctf_yellow_D, 
                                               CUFFT_COMPLEX * frefctf_red_D,
                                               CUFFT_COMPLEX * Fimg_shift_D,
                                               DOUBLE* exp_highres_Xi2_imgs_D,
                                               DOUBLE* Minvsigma2_D,
                                               int* valid_orient_trans_index_D,
                                               int* valid_orient_prefix_sum,
                                               int exp_nr_particles,
                                               int exp_nr_orients,
                                               int exp_nr_trans,
                                               int exp_nr_oversampled_rot,
                                               int exp_nr_oversampled_trans,
                                               int nr_valid_orients,
                                               int diff_xdim,
                                               int image_size,
                                               int  nr_valid_orient_trans)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(exp_nr_oversampled_rot * exp_nr_oversampled_trans * nr_valid_orient_trans, 1, 1);
	calculate_diff2_do_squared_difference_mask_kernel <<< dimGrid, dimBlock>>>(diff2_D,
	                                                                      frefctf_yellow_D,
	                                                                      frefctf_red_D,
	                                                                      Fimg_shift_D,
	                                                                      exp_highres_Xi2_imgs_D,
	                                                                      Minvsigma2_D,
	                                                                      valid_orient_trans_index_D,
	                                                                      valid_orient_prefix_sum,
	                                                                      exp_nr_particles,
	                                                                      exp_nr_orients,
	                                                                      exp_nr_trans,
	                                                                      exp_nr_oversampled_rot,
	                                                                      exp_nr_oversampled_trans,
	                                                                      nr_valid_orients,
	                                                                      diff_xdim,
	                                                                      image_size);
}



__global__ void calculate_diff2_do_squared_difference_pass0_mask_kernel(DOUBLE* diff2_D,
                                                                   const  CUFFT_COMPLEX * __restrict__  frefctf_yellow_D,
                                                                   const  CUFFT_COMPLEX * __restrict__  frefctf_red_D,
                                                                   const  CUFFT_COMPLEX * __restrict__  Fimg_shift_D,
                                                                   const  DOUBLE* __restrict__  exp_highres_Xi2_imgs_D,
                                                                   const  DOUBLE* __restrict__  Minvsigma2_D,
                                                                   const  int* __restrict__ valid_orient_trans_index_D,
                                                                   int exp_nr_particles,
                                                                   int exp_nr_orients,
                                                                   int exp_nr_trans,
                                                                   int exp_nr_oversampled_rot,
                                                                   int exp_nr_oversampled_trans,
                                                                   int nr_valid_orients,
                                                                   int diff_xdim,
                                                                   int image_size,
                                                                   int nr_trans_loop,
                                                                   int nr_trans_stride)

{
	DOUBLE* diff2_array = (DOUBLE*) &diff_array;

	int part_id = blockIdx.y;
	int bid_x = blockIdx.x;
	int tid = threadIdx.x;

	int orient_id = (valid_orient_trans_index_D[bid_x * (exp_nr_trans * exp_nr_particles) + part_id * exp_nr_trans] % (exp_nr_orients * exp_nr_trans)) / (exp_nr_trans);
	int ref_image_offset = (part_id * nr_valid_orients + bid_x) * image_size;

	const  CUFFT_COMPLEX * __restrict__ thisthread_Frefctf_yellow_D = &frefctf_yellow_D[ref_image_offset];
	const  CUFFT_COMPLEX * __restrict__ thisthread_Frefctf_red_D = &frefctf_red_D[ref_image_offset];
	const  DOUBLE* __restrict__ thisthread_Minvsigma2_D = &Minvsigma2_D[part_id * image_size];
	DOUBLE refctf_yellow_real, refctf_yellow_img , minvsigma2;
	DOUBLE refctf_red_real, refctf_red_img;
	DOUBLE Xi2_imgs = exp_highres_Xi2_imgs_D[part_id];
	if (nr_trans_loop == 1)
	{
		DOUBLE diff2_real = 0., diff2_imag = 0.;
		int exp_trans_offset = part_id * exp_nr_trans ;
		for (int trans_id = 0; trans_id < exp_nr_trans; trans_id++)
		{
			diff2_array[tid + trans_id * blockDim.x ] = 0.;
		}

		for (int i = tid; i < image_size; i += blockDim.x)
		{
			refctf_yellow_real = thisthread_Frefctf_yellow_D[i].x;
			refctf_yellow_img = thisthread_Frefctf_yellow_D[i].y;
			refctf_red_real = thisthread_Frefctf_red_D[i].x;
			refctf_red_img = thisthread_Frefctf_red_D[i].y;
			minvsigma2 = thisthread_Minvsigma2_D[i];
#pragma unroll
			for (int trans_id = 0; trans_id < exp_nr_trans; trans_id++)
			{
				diff2_real = refctf_yellow_real + refctf_red_real - Fimg_shift_D[(exp_trans_offset + trans_id) * image_size + i].x;
				diff2_imag = refctf_yellow_img + refctf_red_img - Fimg_shift_D[(exp_trans_offset + trans_id) * image_size + i].y;
				diff2_array[tid + trans_id * blockDim.x ] += (diff2_real * diff2_real + diff2_imag * diff2_imag) * 0.5 * minvsigma2;
			}
		}
		__syncthreads();

		for (int s = blockDim.x >> 1; s > 0; s >>= 1)
		{
			if (tid < s)
			{
#pragma unroll
				for (int trans_id = 0; trans_id < exp_nr_trans; trans_id++)
				{
					diff2_array[tid + trans_id * blockDim.x] += diff2_array[tid + trans_id * blockDim.x + s];
				}
			}
			__syncthreads();
		}

		if (tid < exp_nr_trans)
		{
			diff2_D[part_id * diff_xdim + orient_id * exp_nr_trans + tid] = diff2_array[tid * blockDim.x] + Xi2_imgs * 0.5;
		}
	}
	else
	{
		for (int j = 0; j < nr_trans_loop; j++)
		{
			int nr_trans = (j < (nr_trans_loop - 1)) ? nr_trans_stride : (exp_nr_trans - nr_trans_stride * j);
			DOUBLE diff2_real = 0., diff2_imag = 0.;
			int exp_trans_offset = part_id * exp_nr_trans + nr_trans_stride * j;
			for (int trans_id = 0; trans_id < nr_trans; trans_id++)
			{
				diff2_array[tid + trans_id * blockDim.x ] = 0.;
			}
			for (int i = tid; i < image_size; i += blockDim.x)
			{
			refctf_yellow_real = thisthread_Frefctf_yellow_D[i].x;
			refctf_yellow_img = thisthread_Frefctf_yellow_D[i].y;
			refctf_red_real = thisthread_Frefctf_red_D[i].x;
			refctf_red_img = thisthread_Frefctf_red_D[i].y;
				minvsigma2 = thisthread_Minvsigma2_D[i];
#pragma unroll
				for (int trans_id = 0; trans_id < nr_trans; trans_id++)
				{
					diff2_real = refctf_yellow_real + refctf_red_real - Fimg_shift_D[(exp_trans_offset + trans_id) * image_size + i].x;
					diff2_imag = refctf_yellow_img + refctf_red_img - Fimg_shift_D[(exp_trans_offset + trans_id) * image_size + i].y;
					diff2_array[tid + trans_id * blockDim.x ] += (diff2_real * diff2_real + diff2_imag * diff2_imag) * 0.5 * minvsigma2;
				}
			}
			__syncthreads();

			for (int s = blockDim.x >> 1; s > 0; s >>= 1)
			{
				if (tid < s)
				{
					for (int trans_id = 0; trans_id < nr_trans; trans_id++)
					{
						diff2_array[tid + trans_id * blockDim.x] += diff2_array[tid + trans_id * blockDim.x + s];
					}
				}
				__syncthreads();
			}
			if (tid < nr_trans)
			{
				diff2_D[part_id * diff_xdim + orient_id * exp_nr_trans + tid + nr_trans_stride * j] = diff2_array[tid * blockDim.x] + Xi2_imgs * 0.5;
			}
		}
	}

}

void calculate_diff2_do_squared_difference_pass0_mask_gpu(DOUBLE* diff2_D,
                                                     CUFFT_COMPLEX * frefctf_yellow_D, CUFFT_COMPLEX *frefctf_red_D,
                                                     CUFFT_COMPLEX * Fimg_shift_D,
                                                     DOUBLE* exp_highres_Xi2_imgs_D,
                                                     DOUBLE* Minvsigma2_D,
                                                     int* valid_orient_trans_index_D,
                                                     int exp_nr_particles,
                                                     int exp_nr_orients,
                                                     int exp_nr_trans,
                                                     int exp_nr_oversampled_rot,
                                                     int exp_nr_oversampled_trans,
                                                     int nr_valid_orients,
                                                     int diff_xdim,
                                                     int image_size)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid(nr_valid_orients, exp_nr_particles, 1);
	int shared_size;
	int nr_trans_stride = 4;
	if (exp_nr_trans <= 4)
	{
		shared_size = sizeof(DOUBLE) * (BLOCK_SIZE_128 * exp_nr_trans);
	}
	else
	{
		shared_size = sizeof(DOUBLE) * (BLOCK_SIZE_128 * nr_trans_stride);
	}
	int nr_trans_loop = (exp_nr_trans <= 4) ? 1 : ((exp_nr_trans + (nr_trans_stride - 1)) / nr_trans_stride);
	calculate_diff2_do_squared_difference_pass0_mask_kernel <<< dimGrid, dimBlock, shared_size>>>(diff2_D,
	        frefctf_yellow_D,
	        frefctf_red_D,
	        Fimg_shift_D,
	        exp_highres_Xi2_imgs_D,
	        Minvsigma2_D,
	        valid_orient_trans_index_D,
	        exp_nr_particles,
	        exp_nr_orients,
	        exp_nr_trans,
	        exp_nr_oversampled_rot,
	        exp_nr_oversampled_trans,
	        nr_valid_orients,
	        diff_xdim,
	        image_size,
	        nr_trans_loop,
	        nr_trans_stride);


}


void __global__ apply_CTF_kernel(CUFFT_COMPLEX *Fref_all_shift_D, DOUBLE *exp_Fctf_D, int exp_nr_images, int image_size)
{
	int bid_x = blockIdx.x;
	int tid = threadIdx.x;
	int global_id = tid + bid_x*blockDim.x;

	if(global_id < exp_nr_images* image_size)
	{
		Fref_all_shift_D[global_id].x *= exp_Fctf_D[global_id];
		Fref_all_shift_D[global_id].y *= exp_Fctf_D[global_id];
	}

}
void apply_CTF_gpu(CUFFT_COMPLEX *Fref_all_shift_D, DOUBLE *exp_Fctf_D, int exp_nr_images, int image_size)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((exp_nr_images*image_size+BLOCK_SIZE_128-1)/BLOCK_SIZE_128, 1);
	apply_CTF_kernel<<<dimGrid, dimBlock>>>(Fref_all_shift_D, exp_Fctf_D, exp_nr_images, image_size);
	//std::cout<< "sub project end!!!"<< std::endl;
}


void __global__ sub_Yellow_project_kernel(DOUBLE *image_red_D, DOUBLE *project_imgs_D, int exp_nr_images, int image_size)
{
	int bid_x = blockIdx.x;
	int tid = threadIdx.x;
	int global_id = tid + bid_x*blockDim.x;

	if(global_id < exp_nr_images* image_size)
	{
		image_red_D[global_id] -= project_imgs_D[global_id];
	}

}

void sub_Yellow_project_gpu(DOUBLE* image_red_D, DOUBLE *project_imgs_D, int exp_nr_images,  
								  int xdim, int ydim)
{
	int image_size = xdim*ydim;
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((exp_nr_images*image_size+BLOCK_SIZE_128-1)/BLOCK_SIZE_128, 1);
	sub_Yellow_project_kernel<<<dimGrid, dimBlock>>>(image_red_D, project_imgs_D, exp_nr_images, image_size);

}
	


