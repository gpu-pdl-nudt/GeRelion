/***************************************************************************
 *
 * Author : "Huayou SU, Wen WEN, Xiaoli DU, Dongsheng LI"
 * Parallel and Distributed Processing Laboratory of NUDT
 * Author : "Maofu LIAO"
 * Department of Cell Biology, Harvard Medical School
 *
 * This file is the GPU program for Projector, 
 * including the kernels and host side program.
 * We implemented the key function for computing the Fourier Transform
 * The functions with suffix "_gpu" are simular with 
 * the CPU implementaion in file Projector.cpp
 * Some of the data structure and aux functions are from Relion
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


#include "src/projector.h"
//#define DEBUG

__device__ double atomicAdd_double_2(double* address, double val)
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

extern void compare_CPU_GPU(Complex* in_cpu, CUFFT_COMPLEX * in_gpu, int size_);
extern void compare_CPU_GPU(DOUBLE* in_cpu, DOUBLE* in_gpu, int size_);
//template <typename T>
extern void centerFFT_2_gpu(DOUBLE* in, DOUBLE* out, int nr_images, int dim, int xdim, int ydim, int zdim, bool forward);


__global__ void Pad_translated_map_with_zeros_kernel(DOUBLE* vol_in_D, DOUBLE* Mpad_D, int x1dim, int y1dim, int z1dim, int x2dim, int y2dim,  int z2dim,
                                                     int start_x1,  int start_y1,  int start_z1,  int start_x2,  int start_y2,  int start_z2)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	if (global_index >= x1dim * y1dim * z1dim)
	{
		return;
	}
	int i, j, k;
	j = global_index % x1dim + start_x1;
	i = (global_index / x1dim) % y1dim + start_y1;
	k =  global_index / (x1dim * y1dim) + start_z1;

	Mpad_D[(k - start_z2)*x2dim * y2dim + (i - start_y2)*x2dim + (j - start_x2)] = vol_in_D[global_index];
}

void Pad_translated_map_with_zeros_gpu(DOUBLE* vol_in_D, DOUBLE* Mpad_D, int x1dim, int y1dim, int z1dim, int x2dim, int y2dim, int z2dim)
{
	int data_size = (x1dim * y1dim * z1dim);
	int start_x1, start_y1, start_z1, start_x2, start_y2, start_z2;
	start_x1 = FIRST_XMIPP_INDEX(x1dim);
	start_y1 = FIRST_XMIPP_INDEX(y1dim);
	start_z1 = FIRST_XMIPP_INDEX(z1dim);
	start_x2 = FIRST_XMIPP_INDEX(x2dim);
	start_y2 = FIRST_XMIPP_INDEX(y2dim);
	start_z2 = FIRST_XMIPP_INDEX(z2dim);
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((data_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	//printf("%d %d %d %d %d %d %d %d %d %d %d %d\n", x1dim,  y1dim,  z1dim,  x2dim,  y2dim,  z2dim,
	                                                              //start_x1, start_y1, start_z1, start_x2, start_y2, start_z2);

	Pad_translated_map_with_zeros_kernel <<< dimGrid, dimBlock>>>(vol_in_D, Mpad_D, x1dim,  y1dim,  z1dim,  x2dim,  y2dim,  z2dim,
	                                                              start_x1, start_y1, start_z1, start_x2, start_y2, start_z2);
}

__global__ void normalise_model_data_kernel(CUFFT_COMPLEX * Faux_D, CUFFT_COMPLEX * gdata_ptr, DOUBLE* power_spectrum, DOUBLE* counter, DOUBLE normfft, int x1dim, int y1dim, int z1dim, int x2dim, int y2dim,  int z2dim,
                                            int start_x2,  int start_y2,  int start_z2,   int max_r2, int padding_factor)
{
	int global_index =  threadIdx.x + blockIdx.x * blockDim.x;
	if (global_index >= x1dim * y1dim * z1dim)
	{
		return;
	}
	int i, j, k;
	j = global_index % x1dim;
	i = (global_index / x1dim) % y1dim;
	k =  global_index / (x1dim * y1dim);
	int ip, jp, kp;
	jp = j;
	ip = (i < x1dim) ? i : i - y1dim;
	kp = (k < x1dim) ? k : k - z1dim;
	int r2 = kp * kp + ip * ip + jp * jp;

	if (r2 > max_r2)
	{
		return;
	}

	CUFFT_COMPLEX  a;
	a.x = Faux_D[global_index].x * normfft;
	a.y = Faux_D[global_index].y * normfft;
	gdata_ptr[(kp - start_z2)*x2dim * y2dim + (ip - start_y2)*x2dim + jp] = a ;
	// Calculate power spectrum
	int ires = ((sqrt((DOUBLE)r2) / padding_factor) > 0) ? (int)((sqrt((DOUBLE)r2) / padding_factor) + 0.5) : (int)((sqrt((DOUBLE)r2) / padding_factor) - 0.5);

	// Factor two because of two-dimensionality of the complex plane
	DOUBLE power_spectrum_local = (a.x * a.x + a.y * a.y) / 2;
#ifdef FLOAT_PRECISION
	atomicAdd(&(power_spectrum[ires]), (DOUBLE) power_spectrum_local);
	atomicAdd(&(counter[ires]), (DOUBLE) 1.);
#else
	atomicAdd_double_2(&(power_spectrum[ires]), (DOUBLE) power_spectrum_local);
	atomicAdd_double_2(&(counter[ires]), (DOUBLE) 1.);
#endif
}

void normalise_model_data_gpu(CUFFT_COMPLEX * Faux_D, CUFFT_COMPLEX * gdata_ptr, DOUBLE* power_spectrum, DOUBLE* counter, DOUBLE normfft, int padoridim, int xdim, int ydim, int zdim,  int ref_dim, int max_r2, int padding_factor)
{
	int data_size = (padoridim / 2 + 1) * padoridim * (ref_dim == 2 ? 1 : padoridim);
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((data_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	int start_x2 , start_y2, start_z2;
	start_x2 = 0;
	start_y2 =  -(int)((float)(ydim) / 2.0);
	start_z2 = -(int)((float)(zdim) / 2.0);
	normalise_model_data_kernel <<< dimGrid, dimBlock>>>(Faux_D, gdata_ptr, power_spectrum, counter,  normfft, (padoridim / 2 + 1), padoridim, (ref_dim == 2 ? 1 : padoridim),
	                                                     xdim, ydim, zdim,  start_x2 , start_y2, start_z2, max_r2, padding_factor);
}

__global__ void calculate_power_spectrum_kernel(DOUBLE* power_spectrum, DOUBLE* counter,  int size)
{
	int global_index =  threadIdx.x + blockIdx.x * blockDim.x;
	if (global_index >= size)
	{
		return;
	}
	if (counter[global_index] < 1.)
	{
		power_spectrum[global_index] = 0.;
	}
	else
	{
		power_spectrum[global_index] /= counter[global_index];
	}
}
void calculate_power_spectrum_gpu(DOUBLE* power_spectrum, DOUBLE* counter,  int size)
{
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	calculate_power_spectrum_kernel <<< dimGrid, dimBlock>>>(power_spectrum, counter, size);

}
// Fill data array with oversampled Fourier transform, and calculate its power spectrum
void Projector::computeFourierTransformMap_gpu(MultidimArray<DOUBLE>& vol_in, MultidimArray<DOUBLE>& power_spectrum, int current_size, int nr_threads, bool do_gridding)
{

	//MultidimArray<Complex > Faux;
	FourierTransformer transformer;
	// DEBUGGING: multi-threaded FFTWs are giving me a headache?
	DOUBLE normfft;

	// Size of padded real-space volume
	int padoridim = padding_factor * ori_size;

	// Initialize data array of the oversampled transform
	ref_dim = vol_in.getDim();
	DOUBLE* Mpad_D;
	DOUBLE* vol_in_D;

	cudaMalloc((void**) &vol_in_D, vol_in.zyxdim * sizeof(DOUBLE));
	cudaMemcpy(vol_in_D, vol_in.data, vol_in.zyxdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);

	// Make Mpad
	switch (ref_dim)
	{
	case 2:
		normfft = (DOUBLE)(padding_factor * padding_factor);
		cudaMalloc((void**) &Mpad_D, padoridim * padoridim * sizeof(DOUBLE));
		cudaMemset(Mpad_D, 0., padoridim * padoridim * sizeof(DOUBLE));
		break;
	case 3:
		normfft = (DOUBLE)(padding_factor * padding_factor * padding_factor * ori_size);
		cudaMalloc((void**) &Mpad_D, padoridim * padoridim * padoridim * sizeof(DOUBLE));
		cudaMemset(Mpad_D, 0., padoridim * padoridim * padoridim * sizeof(DOUBLE));
		break;
	default:
		REPORT_ERROR("Projector::get2DSlice%%ERROR: Dimension of the data array should be 2 or 3");
	}


	if (do_gridding)
	{
		griddingCorrect_gpu(vol_in_D, vol_in.xdim, vol_in.ydim, vol_in.zdim, interpolator, r_min_nn,  padoridim);
{	cudaError cudaStat = cudaGetLastError(); if (cudaStat != cudaSuccess) { printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat)); exit(EXIT_FAILURE); } } 
	}

	// Pad translated map with zeros
	vol_in.setXmippOrigin();
	
{	cudaError cudaStat = cudaGetLastError(); if (cudaStat != cudaSuccess) { printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat)); exit(EXIT_FAILURE); } } 
	printf("%d %d %d %d\n", vol_in.xdim, vol_in.ydim, vol_in.zdim, padoridim);

	Pad_translated_map_with_zeros_gpu(vol_in_D, Mpad_D, vol_in.xdim, vol_in.ydim, vol_in.zdim, padoridim, padoridim, (ref_dim == 2 ? 1 : padoridim));

	cudaError cudaStat = cudaGetLastError(); if (cudaStat != cudaSuccess) { printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat)); exit(EXIT_FAILURE); } 
	cudaFree(vol_in_D);
	cudaMalloc((void**)&vol_in_D, padoridim * padoridim * (ref_dim == 2 ? 1 : padoridim)*sizeof(DOUBLE));
	centerFFT_2_gpu(Mpad_D, vol_in_D, 1, ref_dim, padoridim, padoridim, (ref_dim == 2 ? 1 : padoridim),  true);

	cudaFree(Mpad_D);

	CUFFT_COMPLEX * Faux_D;
	cudaMalloc((void**)&Faux_D, (padoridim / 2 + 1)*padoridim * (ref_dim == 2 ? 1 : padoridim)*sizeof(CUFFT_COMPLEX ));
	transformer.FourierTransform_gpu(vol_in_D, Faux_D, 1, padoridim, padoridim, (ref_dim == 2 ? 1 : padoridim), true);

	cudaFree(vol_in_D);
	// Resize data array to the right size and initialise to zero
	initZeros(current_size);
	CUFFT_COMPLEX * gdata_ptr;
	cudaMalloc((void**)&gdata_ptr, data.zyxdim * sizeof(CUFFT_COMPLEX ));
	cudaMemset(gdata_ptr, 0., data.zyxdim * sizeof(CUFFT_COMPLEX ));
	// Fill data only for those points with distance to origin less than max_r
	// (other points will be zero because of initZeros() call above
	// Also calculate radial power spectrum
	power_spectrum.initZeros(ori_size / 2 + 1);

	int max_r2 = r_max * r_max * padding_factor * padding_factor;

	DOUBLE* power_spectrum_D;
	DOUBLE* counter_D;
	cudaMalloc((void**)&power_spectrum_D, sizeof(DOUBLE) * (ori_size / 2 + 1));
	cudaMalloc((void**)&counter_D, sizeof(DOUBLE) * (ori_size / 2 + 1));
	cudaMemset(power_spectrum_D, 0., sizeof(DOUBLE) * (ori_size / 2 + 1));
	cudaMemset(counter_D, 0., sizeof(DOUBLE) * (ori_size / 2 + 1));
	normalise_model_data_gpu(Faux_D,
	                         gdata_ptr,
	                         power_spectrum_D,
	                         counter_D,
	                         normfft,
	                         padoridim,
	                         data.xdim,
	                         data.ydim,
	                         data.zdim,
	                         ref_dim,
	                         max_r2,
	                         padding_factor);

	calculate_power_spectrum_gpu(power_spectrum_D, counter_D, (ori_size / 2 + 1));
	cudaMemcpy(power_spectrum.data, power_spectrum_D, sizeof(DOUBLE) * (ori_size / 2 + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(data.data, gdata_ptr, sizeof(CUFFT_COMPLEX )*data.zyxdim, cudaMemcpyDeviceToHost);
	transformer.free_memory_gpu();
	transformer.cleanup();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}
	cudaFree(power_spectrum_D);
	cudaFree(counter_D);
	cudaFree(Faux_D);
	cudaFree(gdata_ptr);
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}

}

__global__ void griddingCorrect_kernel(DOUBLE* vol_in, int xdim, int ydim, int zdim, int xinit, int yinit, int zinit, int interpolator, int r_min_nn, int ori_size_padding_factor)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	int i, j, k;
	j = ((global_index % xdim) + xinit);
	i = ((global_index / xdim) % ydim + yinit);
	k = (global_index / (xdim * ydim) + zinit);
	DOUBLE r = sqrt((DOUBLE)(k * k + i * i + j * j));

	if (r == 0. || global_index >= xdim * ydim * zdim)
	{
		return;
	}

	DOUBLE rval = r / (ori_size_padding_factor);
	DOUBLE sinc = sin(PI * rval) / (PI * rval);
	// Interpolation (goes with "interpolator") to go from arbitrary to fine grid
	if (interpolator == NEAREST_NEIGHBOUR && r_min_nn == 0)
	{
		// NN interpolation is convolution with a rectangular pulse, which FT is a sinc function
		vol_in[global_index] /= sinc;
	}
	else if (interpolator == TRILINEAR || (interpolator == NEAREST_NEIGHBOUR && r_min_nn > 0))
	{
		// trilinear interpolation is convolution with a triangular pulse, which FT is a sinc^2 function
		vol_in[global_index] /= sinc * sinc;
	}

}
void Projector::griddingCorrect_gpu(DOUBLE* vol_in, int xdim, int ydim, int zdim, int interpolator, int r_min_nn, int ori_size_padding_factor)
{
	// Correct real-space map by dividing it by the Fourier transform of the interpolator(s)
	int zinit = FIRST_XMIPP_INDEX(zdim);
	int yinit = FIRST_XMIPP_INDEX(ydim);
	int xinit = FIRST_XMIPP_INDEX(xdim);
	int data_size = xdim * ydim * zdim;
	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((data_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	//printf("%x %d %d %d %d %d %d %d %d %d %d\n",  vol_in, data_size, xdim,  ydim,  zdim, xinit, yinit, zinit, interpolator, r_min_nn, ori_size_padding_factor);
{	cudaError cudaStat = cudaGetLastError(); if (cudaStat != cudaSuccess) { printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat)); exit(EXIT_FAILURE); } } 
	griddingCorrect_kernel <<< dimGrid , dimBlock>>>(vol_in,  xdim,  ydim,  zdim, xinit, yinit, zinit, interpolator, r_min_nn, ori_size_padding_factor);
{	cudaError cudaStat = cudaGetLastError(); if (cudaStat != cudaSuccess) { printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat)); exit(EXIT_FAILURE); } } 
}

