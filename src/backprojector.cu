/***************************************************************************
 *
 * Author : "Huayou SU, Wen WEN, Xiaoli DU, Dongsheng LI"
 * Parallel and Distributed Processing Laboratory of NUDT
 * Author : "Maofu LIAO"
 * Department of Cell Biology, Harvard Medical School
 *
 * This file is the GPU program for backproject, 
 * including the kernels and host side program.
 * We implemented the key function  reconstruct with GPU, named reconstruct_gpu.
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


#include "src/backprojector.h"
#include "src/math_function.h"

__constant__ DOUBLE __L_array [100 * 4 * 4];
__constant__ DOUBLE __R_array [100 * 4 * 4];

#ifndef FLOAT_PRECISION
__device__ double atomicAdd_double(double* address, double val)
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
void compare_CPU_GPU(Complex* in_cpu, CUFFT_COMPLEX * in_gpu, int size_)
{
	Complex* in_gpu_H;
	in_gpu_H = (Complex*) malloc(size_ * sizeof(Complex));
	cudaMemcpy(in_gpu_H, in_gpu, size_ * sizeof(Complex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size_; i++)
	{
		if (abs(in_cpu[i] - in_gpu_H[i]) > 0.00000001)
		{
			std::cout << "Resuls Error at real : " << i << "  " << in_cpu[i].real << "  " << in_gpu_H[i].real << std::endl;
			std::cout << "Resuls Error at imag: " << i << "  " << in_cpu[i].imag << "  " << in_gpu_H[i].imag << std::endl;
			for (int j = i; j < i + 40; j++)
			{
				std::cout << "Resuls Error at real : " << j << "  " << in_cpu[j].real << "  " << in_gpu_H[j].real << std::endl;
				std::cout << "Resuls Error at imag: " << j << "  " << in_cpu[j].imag << "  " << in_gpu_H[j].imag << std::endl;
			}

			REPORT_ERROR("ERROR: in_cpu[i]!=in_gpu_H[i] ");

		}

	}
	free(in_gpu_H);
}

void compare_CPU_GPU(DOUBLE* in_cpu, DOUBLE* in_gpu, int size_)
{
	DOUBLE* in_gpu_H;
	in_gpu_H = (DOUBLE*) malloc(size_ * sizeof(DOUBLE));
	cudaMemcpy(in_gpu_H, in_gpu, size_ * sizeof(DOUBLE), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size_; i++)
	{
		if (abs(in_cpu[i] - in_gpu_H[i]) > 0.00000001)
		{
			std::cout << "Resuls Error at: " << i << "  " << in_cpu[i] << "  " << in_gpu_H[i] << std::endl;
			REPORT_ERROR("ERROR: in_cpu[i]!=in_gpu_H[i] ");
		}
	}
	free(in_gpu_H);
}

__global__ void update_tau2_with_fsc_kernel(const  DOUBLE* __restrict__ sigma2_D, DOUBLE* fsc_D, DOUBLE* tau2_D, DOUBLE* data_vs_prior_D, int data_size, bool is_whole_instead_of_half)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	if (global_index >= data_size)
	{
		return;
	}
	DOUBLE myfsc = (0.001 >= fsc_D[global_index]) ? (0.001) : fsc_D[global_index];
	if (is_whole_instead_of_half)
	{
		// Factor two because of twice as many particles
		// Sqrt-term to get 60-degree phase errors....
		myfsc = sqrt(2. * myfsc / (myfsc + 1.));
	}
	myfsc = (myfsc >= 0.999) ? (0.999) : myfsc;
	DOUBLE myssnr = myfsc / (1. - myfsc);
	DOUBLE fsc_based_tau = myssnr * sigma2_D[global_index];
	tau2_D[global_index] = fsc_based_tau;
	// data_vs_prior is merely for reporting: it is not used for anything in the reconstruction
	data_vs_prior_D[global_index] = myssnr;
}
void update_tau2_with_fsc_gpu(DOUBLE* sigma2_D, DOUBLE* fsc_D, DOUBLE* tau2_D, DOUBLE* data_vs_prior_D, int data_size, bool is_whole_instead_of_half)
{
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((data_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);

	update_tau2_with_fsc_kernel <<< gridDim, blockDim>>>(sigma2_D, fsc_D, tau2_D, data_vs_prior_D, data_size, is_whole_instead_of_half);

}

__global__ void Applymap_additional_to_weight_kernel(DOUBLE* weight_D, DOUBLE* tau2_D, DOUBLE* data_vs_prior_D, DOUBLE* counter_D,
                                                     int max_r2, int xdim, int ydim, int zdim,
                                                     int padding_factor, DOUBLE oversampling_correction, DOUBLE tau2_fudge, bool update_tau2_with_fsc, int minres_map)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	int i, j, k;
	int ip, jp, kp;
	j = global_index % xdim ;
	i = (global_index / xdim) % ydim;
	k =  global_index / (xdim * ydim);

	jp = j;
	ip = (i < xdim) ? i : (i - ydim);
	kp = (k < xdim) ? k : (k - zdim);
	int r2 = (kp * kp + ip * ip + jp * jp);
	if (global_index >= (xdim * ydim * zdim) || r2 >= max_r2)
	{
		return;
	}
	DOUBLE invtau2;
	int ires = ((sqrt((DOUBLE)r2) / padding_factor) > 0) ? (int)((sqrt((DOUBLE)r2) / padding_factor) + 0.5) : (int)((sqrt((DOUBLE)r2) / padding_factor) - 0.5);

	DOUBLE invw = weight_D[global_index];
	// We consider that the values of tau2 will not be negative
	invtau2 = (tau2_D[ires] > 0.) ? (1. / (oversampling_correction * tau2_fudge * tau2_D[ires])) : (1. / (0.001 * invw));

	if (!update_tau2_with_fsc)
	{
#ifdef FLOAT_PRECISION
		atomicAdd(&(data_vs_prior_D[ires]), (DOUBLE) invw / invtau2);
#else
		atomicAdd_double(&(data_vs_prior_D[ires]), (DOUBLE) invw / invtau2);
#endif
	}
#ifdef FLOAT_PRECISION
		atomicAdd(&(counter_D[ires]), (DOUBLE) 1.0);
#else
		atomicAdd_double(&(counter_D[ires]), (DOUBLE) 1.0);
#endif
	if (ires >= minres_map)
	{
		weight_D[global_index] = invw + invtau2;
	}
}
void Applymap_additional_to_weight_gpu(DOUBLE* weight_D, DOUBLE* tau2_D, DOUBLE* data_vs_prior_D, DOUBLE* counter_D,
                                       int max_r2, int xdim, int ydim, int zdim,
                                       int padding_factor, DOUBLE oversampling_correction, DOUBLE tau2_fudge, bool update_tau2_with_fsc, int minres_map)

{
	int model_size = xdim * ydim * zdim;
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((model_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	Applymap_additional_to_weight_kernel <<< gridDim, blockDim>>>(weight_D, tau2_D, data_vs_prior_D, counter_D,
	                                                              max_r2,  xdim,  ydim, zdim,
	                                                              padding_factor,  oversampling_correction,  tau2_fudge,  update_tau2_with_fsc,  minres_map);


}

__global__ void Average_data_vs_prior_kernel(DOUBLE* data_vs_prior_D, const DOUBLE* __restrict__ counter_D, int data_size, int r_max)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= data_size)
	{
		return;
	}

	if (i > r_max)
	{
		data_vs_prior_D[i] = 0.;
	}
	else if (counter_D[i] < 0.001)
	{
		data_vs_prior_D[i] = 999.;
	}
	else
	{
		data_vs_prior_D[i] /= counter_D[i];
	}
}
void Average_data_vs_prior_gpu(DOUBLE* data_vs_prior_D, DOUBLE* counter_D, int data_size, int r_max)
{
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((data_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	Average_data_vs_prior_kernel <<< gridDim,  blockDim>>>(data_vs_prior_D, counter_D, data_size, r_max) ;
}

__global__ void do_normalise_data_kernel(CUFFT_COMPLEX * data_D, int data_size, DOUBLE normalise_value)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i > data_size)
	{
		return;
	}
	data_D[i].x = data_D[i].x / normalise_value;
	data_D[i].y = data_D[i].y / normalise_value;
}
__global__ void do_normalise_weight_kernel(DOUBLE* weight_D, int weight_size, DOUBLE normalise_value)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i > weight_size)
	{
		return;
	}
	weight_D[i] = weight_D[i] / normalise_value;
}
void do_normalise_weight_data_gpu(DOUBLE* weight_D, CUFFT_COMPLEX * data_D, int weight_size, int data_size, DOUBLE normalise_value)
{
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((data_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	do_normalise_data_kernel <<< gridDim , blockDim>>>(data_D, data_size, normalise_value);

	dim3 gridDim2((weight_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	do_normalise_weight_kernel <<< gridDim2 , blockDim>>>(weight_D, weight_size, normalise_value);
}

__global__ void init_Fnewweight_kernel(double* Fnewweight_D, int xdim, int ydim, int zdim, int my_rmax2)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (global_index > xdim * ydim * zdim)
	{
		return;
	}

	int i, j, k;
	int ip, jp, kp;
	j = global_index % xdim ;
	i = (global_index / xdim) % ydim;
	k = global_index / (xdim * ydim);

	jp = j;
	ip = (i < xdim) ? i : (i - ydim);
	kp = (k < xdim) ? k : (k - zdim);
	int r2 = kp * kp + ip * ip + jp * jp;
	if (r2 < my_rmax2)
	{
		Fnewweight_D[global_index] = 1.0;
	}
	else
	{
		Fnewweight_D[global_index] = 0.0;
	}
}
void init_Fnewweight_gpu(double* Fnewweight_D, int xdim, int ydim, int zdim, int my_rmax2)
{
	int data_size = xdim * ydim * zdim;
	dim3  blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((data_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);

	init_Fnewweight_kernel <<< gridDim, blockDim>>>(Fnewweight_D, xdim, ydim, zdim, my_rmax2);
}
__global__ void calculate_sigma2_kernel(const DOUBLE* __restrict__ weight_D, DOUBLE* sigma2_D, DOUBLE* counter_D, int padding_factor, DOUBLE oversampling_correction,
                                        int xdim, int ydim, int zdim, int max_r2, int size_sigma)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	int i, j, k; // x, y, z;
	int ip, jp, kp;
	j = global_index % xdim;
	i = (global_index / xdim) % ydim;
	k =  global_index / (xdim * ydim);

	jp = j;
	ip = (i < xdim) ? i : (i - ydim);
	kp = (k < xdim) ? k : (k - zdim);
	int ires = (kp * kp + ip * ip + jp * jp);
	if (global_index >= (xdim * ydim * zdim) || ires >= max_r2)
	{
		return;
	}
	int ires_id = ((sqrt((DOUBLE)ires) / padding_factor) > 0) ? (int)((sqrt((DOUBLE)ires) / padding_factor) + 0.5) : (int)((sqrt((DOUBLE)ires) / padding_factor) - 0.5);
	DOUBLE invw = oversampling_correction * weight_D[global_index];
#ifdef FLOAT_PRECISION
	atomicAdd(&(sigma2_D[ires_id]), (DOUBLE) invw);
	atomicAdd(&(counter_D[ires_id]), (DOUBLE) 1.);
#else
	atomicAdd_double(&(sigma2_D[ires_id]), (DOUBLE) invw);
	atomicAdd_double(&(counter_D[ires_id]), (DOUBLE) 1.);
#endif
	
}
__global__ void average_Sigma_kernel(DOUBLE* sigma2_D, DOUBLE* counter_D, int size_sigma)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	if (global_index >= size_sigma)
	{
		return;
	}
	sigma2_D[global_index] = (sigma2_D[global_index] > 1e-10) ? (counter_D[global_index] / sigma2_D[global_index]) : (0.);
}
void calculate_sigma2_gpu(DOUBLE* weight_D , DOUBLE* sigma2_D, DOUBLE* counter_D, int padding_factor, DOUBLE oversampling_correction,
                          int xdim, int ydim, int zdim, int max_r2, int size_sigma)
{
	int model_size = xdim * ydim * zdim;
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((model_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	calculate_sigma2_kernel <<< gridDim, blockDim>>>(weight_D, sigma2_D, counter_D,  padding_factor, oversampling_correction,
	                                                 xdim,  ydim,  zdim,  max_r2,  size_sigma);

	dim3 blockDim2(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim2((size_sigma + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	average_Sigma_kernel <<< gridDim2 , blockDim2>>>(sigma2_D, counter_D, size_sigma);
}
__global__ void init_Fconv_kernel(CUFFT_COMPLEX * Fconv_D, double* Fnewweight_D, DOUBLE* Fweight_D, int model_size)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (global_index >= model_size)
	{
		return;
	}
	Fconv_D[global_index].x = Fnewweight_D[global_index] * Fweight_D[global_index];
	Fconv_D[global_index].y = 0.;
}

void init_Fconv_gpu(CUFFT_COMPLEX * Fconv_D, double* Fnewweight_D, DOUBLE* Fweight_D, long int model_size)
{
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((model_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);

	init_Fconv_kernel <<< gridDim, blockDim>>>(Fconv_D, Fnewweight_D, Fweight_D, model_size);

}
__global__ void Multi_by_FT_tab_kernel(DOUBLE* Mconv_D, const DOUBLE* __restrict__ tab_ftblob_D,  DOUBLE normftblob,  DOUBLE sampling, int pad_size, int padhdim, int ori_size_padding_factor, int padding_factor, int tab_size, bool do_mask)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	if (global_index >= (pad_size * pad_size * pad_size))
	{
		return;
	}

	int i, j, k;
	int ip, jp, kp;
	j = global_index % pad_size;
	i = (global_index / pad_size) % pad_size;
	k =  global_index / (pad_size * pad_size);
	kp = (k < padhdim) ? k : k - pad_size;
	ip = (i < padhdim) ? i : i - pad_size;
	jp = (j < padhdim) ? j : j - pad_size;
	DOUBLE rval = sqrt((DOUBLE)(kp * kp + ip * ip + jp * jp)) / (ori_size_padding_factor);
	if (do_mask && rval > 1. / (2. * padding_factor))
	{
		Mconv_D[global_index] = 0.;
	}
	else
	{
		int idx = (int)(abs(rval) / sampling);
		if (idx >= tab_size)
		{
			Mconv_D[global_index] = 0.;
		}
		else
		{
			Mconv_D[global_index] *= (tab_ftblob_D[idx] / normftblob);
		}
	}
}
void Multi_by_FT_tab_gpu(DOUBLE* Mconv_D,  DOUBLE* tab_ftblob_D,  DOUBLE normftblob,   DOUBLE sampling, int pad_size, int padhdim, int ori_size_padding_factor, int padding_factor, int tab_size, bool do_mask)
{
	int model_size = pad_size * pad_size * pad_size;
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((model_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	Multi_by_FT_tab_kernel <<< gridDim, blockDim>>>(Mconv_D, tab_ftblob_D, normftblob,  sampling, pad_size, padhdim, ori_size_padding_factor, padding_factor, tab_size, do_mask);
}

__global__ void update_Fconv_kernel(const CUFFT_COMPLEX * __restrict__ Fconv_D, double* Fnewweight_D, int xdim, int ydim, int zdim, int max_r2)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;
	int i, j, k;
	int ip, jp, kp;
	j = global_index % xdim;
	i = (global_index / xdim) % ydim;
	k =  global_index / (xdim * ydim);
	kp = (k < xdim) ? k : k - zdim;
	ip = (i < xdim) ? i : i - ydim;
	jp = j;
	int r2 = kp * kp + ip * ip + jp * jp;
	if (global_index >= (xdim * ydim * zdim) || r2 >= max_r2)
	{
		return;
	}
	DOUBLE w = sqrt(Fconv_D[global_index].x * Fconv_D[global_index].x + Fconv_D[global_index].y * Fconv_D[global_index].y);
	w = (((1e-6) >= w) ? (1e-6) : (w));
	Fnewweight_D[global_index] =Fnewweight_D[global_index]/w;

}
void update_Fconv_gpu(CUFFT_COMPLEX * Fconv_D, double* Fnewweight_D, int xdim, int ydim, int zdim, int max_r2)
{
	int model_size = xdim * ydim * zdim;
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((model_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	update_Fconv_kernel <<< gridDim , blockDim>>>(Fconv_D, Fnewweight_D, xdim, ydim, zdim, max_r2);
}



__global__ void centerFFT_2_kernel(DOUBLE* in, DOUBLE* out, int xdim, int ydim, int xshift, int yshift)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;
	int i, j;
	j = global_index % xdim;
	i = (global_index / xdim) % ydim;
	int jp = j + xshift;
	int ip = i + yshift;
	int posy = (ip >= ydim) ? (ip - ydim) : ((ip < 0) ? (ip + ydim) : ip);
	int posx = (jp >= xdim) ? (jp - xdim) : ((jp < 0) ? (jp + xdim) : jp);
	if (global_index >= xdim * ydim )
	{
		return;
	}

	out[posx + posy * xdim + blockIdx.y * xdim * ydim ] = in[j + i * xdim  + blockIdx.y * xdim * ydim];

}

//template <typename T>
__global__ void centerFFT_3_kernel(DOUBLE* in, DOUBLE* out, int xdim, int ydim, int zdim, int xshift, int yshift, int zshift)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;
	int i, j, k;
	j = global_index % xdim;
	i = (global_index / xdim) % ydim;
	k =  global_index / (xdim * ydim);
	int jp = j + xshift;
	int ip = i + yshift;
	int kp = k + zshift;
	int posz = (kp >= zdim) ? (kp - zdim) : ((kp < 0) ? (kp + zdim) : kp);
	int posy = (ip >= ydim) ? (ip - ydim) : ((ip < 0) ? (ip + ydim) : ip);
	int posx = (jp >= xdim) ? (jp - xdim) : ((jp < 0) ? (jp + xdim) : jp);
	if (global_index >= xdim * ydim * zdim)
	{
		return;
	}

	out[posx + posy * xdim + posz * xdim * ydim + blockIdx.y * xdim * ydim * zdim] = in[j + i * xdim + k * xdim * ydim + blockIdx.y * xdim * ydim * zdim];

}

//template <typename T>
void centerFFT_2_gpu(DOUBLE* in, DOUBLE* out, int nr_images, int dim, int xdim, int ydim, int zdim, bool forward)
{
	int size = xdim * ydim * zdim;

	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, nr_images, 1);
	int xshift = (int)(xdim / 2);
	int yshift = (int)(ydim / 2);
	int zshift = (int)(zdim / 2);
	if (!forward)
	{
		xshift = -xshift;
		yshift = -yshift;
		zshift = -zshift;
	}
	if(dim == 2)
		centerFFT_2_kernel <<< dimGrid, dimBlock>>>(in, out, xdim, ydim, xshift, yshift);
	else if(dim ==3)
		centerFFT_3_kernel <<< dimGrid, dimBlock>>>(in, out, xdim, ydim, zdim, xshift, yshift, zshift);
}

// Explicit instantiation
//template void centerFFT_2_gpu<DOUBLE>(DOUBLE* in, DOUBLE* out, int nr_images, int dim, int xdim, int ydim, int zdim, bool forward);
//template void centerFFT_2_gpu<float>(float* in, float* out, int nr_images, int dim, int xdim, int ydim, int zdim, bool forward);

//template <typename T>
__global__ void window_kernel(DOUBLE* in, DOUBLE* out, DOUBLE normfft, int start_x1,  int start_y1,  int start_z1,  int start_x2,  int start_y2,  int start_z2,
                              int x1dim, int y1dim, int z1dim, int x2dim, int y2dim, int z2dim,
                              DOUBLE init_value, int n)
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

	if (j >= start_x2 && j <= (start_x2 + x2dim - 1) && i >= start_y2 && i <= (start_y2 + y2dim - 1)  && k >= start_z2 && k <= (start_z2 + z2dim - 1))
	{
		out[global_index] = in[(k - start_z2) * x2dim * y2dim + (i - start_y2) * x2dim + (j - start_x2) + n * x2dim * y2dim * z2dim] / normfft;
	}
	else
	{
		out[global_index] = init_value / normfft;
	}

}
//template <typename T>
void window_gpu(DOUBLE* in, DOUBLE* out,  DOUBLE normfft, int start_x1,  int start_y1,  int start_z1,  int start_x2,  int start_y2,  int start_z2,
                int x1dim, int y1dim, int z1dim, int x2dim, int y2dim, int z2dim,
                DOUBLE init_value = 0, int n = 0)
{
	int data_size = (x1dim * y1dim * z1dim);

	dim3 dimBlock(BLOCK_SIZE_128, 1, 1);
	dim3 dimGrid((data_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	window_kernel <<< dimGrid , dimBlock>>>(in, out, normfft, start_x1, start_y1, start_z1, start_x2,   start_y2,   start_z2,
	                                        x1dim,  y1dim,  z1dim,  x2dim,  y2dim,  z2dim,
	                                        init_value,  n) ;

}
//template void window_gpu<DOUBLE>(DOUBLE* in, DOUBLE* out, DOUBLE normfft,  int start_x1,  int start_y1,  int start_z1,  int start_x2,  int start_y2,  int start_z2,
//                                 int x1dim, int y1dim, int z1dim, int x2dim, int y2dim, int z2dim,
//                                DOUBLE init_value = 0, int n = 0);
//template void window_gpu<float>(float* in, float* out, DOUBLE normfft, int start_x1,  int start_y1,  int start_z1,  int start_x2,  int start_y2,  int start_z2,
//                                int x1dim, int y1dim, int z1dim, int x2dim, int y2dim, int z2dim,
//                                float init_value = 0, int n = 0);

static __global__ void softMaskOutsideMap_new_kernel(DOUBLE* vol, DOUBLE radius, DOUBLE cosine_width, DOUBLE* Mnoise, DOUBLE radius_p, int xdim, int ydim, int zdim, int xinit, int yinit, int zinit)
{
	int offset;
	int tid = threadIdx.x;

	int image_size = zdim * ydim * xdim;

	if (tid >= image_size)
	{
		return;
	}
	long int kp, ip, jp;
	offset = blockIdx.x * image_size;
	DOUBLE r, raisedcos;

	__shared__ DOUBLE sum_bg[512];
	__shared__ DOUBLE sum[512];
	sum_bg[tid] = 0;
	sum[tid] = 0;

	if (Mnoise == NULL)
	{
		for (int i = tid; i < image_size; i += blockDim.x)
		{
			jp = ((i % xdim) + xinit);
			ip = ((i / xdim) % ydim + yinit);
			kp = (i / (xdim * ydim) + zinit);
			r = sqrt((DOUBLE)(kp * kp + ip * ip + jp * jp));

			if (r < radius)
				;
			else if (r > radius_p)
			{
				sum[tid]    += 1.;
				sum_bg[tid] += vol[offset + (kp - zinit) * xdim * ydim + (ip - yinit) * xdim + (jp - xinit)];
			}
			else
			{
				raisedcos = 0.5 + 0.5 * cos(PI * (radius_p - r) / cosine_width);
				sum[tid] += raisedcos;
				sum_bg[tid] += raisedcos * vol[offset + (kp - zinit) * xdim * ydim + (ip - yinit) * xdim + (jp - xinit)];
			}
		}
	}
	////////////TODO: refuction the sum of sum and sum_bg
	// do reduction in shared mem
	__syncthreads();
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
	for (int i = tid; i < image_size; i += blockDim.x)
	{
		jp = (i % xdim + xinit);
		ip = (i / xdim) % ydim + yinit;
		kp = (i / (xdim * ydim) + zinit);
		r = sqrt((DOUBLE)(kp * kp + ip * ip + jp * jp));
		if (r > radius_p && r >= radius)
		{
			vol[offset + (kp - zinit)*xdim * ydim + (ip - yinit)*xdim + (jp - xinit)] = (Mnoise == NULL) ? sum_bg[0] : Mnoise[offset + (kp - zinit) * xdim * ydim + (ip - yinit) * xdim + (jp - xinit)];
		}
		else if (r <= radius_p && r >= radius)
		{
			raisedcos = 0.5 + 0.5 * cos(PI * (radius_p - r) / cosine_width);
			DOUBLE add = (Mnoise == NULL) ?  sum_bg[0] : Mnoise[offset + (kp - zinit) * xdim * ydim + (ip - yinit) * xdim + (jp - xinit)];
			vol[offset + (kp - zinit)*xdim * ydim + (ip - yinit)*xdim + (jp - xinit)] = (1 - raisedcos) * vol[offset + (kp - zinit) * xdim * ydim + (ip - yinit) * xdim + (jp - xinit)] + raisedcos * add;
		}
	}

}

//template <typename T>
void softMaskOutsideMap_new_gpu(DOUBLE* vol, DOUBLE radius, DOUBLE cosine_width, DOUBLE* Mnoise, int nr_images, int xdim, int ydim, int zdim)
{
	int zinit = FIRST_XMIPP_INDEX(zdim);
	int yinit = FIRST_XMIPP_INDEX(ydim);
	int xinit = FIRST_XMIPP_INDEX(xdim);

	DOUBLE radius_p;
	if (radius < 0)
	{
		radius = (DOUBLE)xdim / 2.;
	}
	radius_p = radius + cosine_width;

	dim3 blockDim(512, 1, 1);
	dim3 gridDim(nr_images, 1, 1);
	int shared_mem_size = 512 * sizeof(DOUBLE) * 2;

	softMaskOutsideMap_new_kernel <<< gridDim, blockDim>>>(vol,  radius,  cosine_width, Mnoise, radius_p, xdim, ydim, zdim, xinit, yinit, zinit);

}

//template void softMaskOutsideMap_new_gpu<DOUBLE>(DOUBLE* vol, DOUBLE radius, DOUBLE cosine_width, DOUBLE* Mnoise, int nr_images, int xdim, int ydim, int zdim);

extern __shared__ DOUBLE  spectrum_count[ ];
__global__ void update_tau2_kernel(CUFFT_COMPLEX * Fconv_D, DOUBLE* tau2_D, DOUBLE tau2_fudge, int xdim, int ydim, int zdim, int data_vs_prior_size, int ori_size, DOUBLE  normfft)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;
	DOUBLE* spectrum = (DOUBLE*)spectrum_count;
	DOUBLE* count = (DOUBLE*)&spectrum[ori_size];

	if (threadIdx.x < ori_size)
	{
		spectrum[threadIdx.x] = 0.;
		count[threadIdx.x] = 0.;
	}
	int i, j, k;
	int jp, ip, kp;
	for (global_index = threadIdx.x; global_index < xdim * ydim * zdim; global_index += blockDim.x)
	{
		j = ((global_index % xdim));
		i = ((global_index / xdim) % ydim);
		k = (global_index / (xdim * ydim));
		kp = (k < xdim) ? k : k - zdim;
		ip = (i < xdim) ? i : i - ydim;
		jp = j; // (j < padhdim) ? j : j - pad_size;
		int r2 = kp * kp + ip * ip + jp * jp;
		int idx = (sqrt((DOUBLE)r2)) > 0 ? (int)((sqrt((DOUBLE)r2)) + 0.5) : (int)((sqrt((DOUBLE)r2)) - 0.5);
		DOUBLE normmal = Fconv_D[global_index].x * Fconv_D[global_index].x + Fconv_D[global_index].y * Fconv_D[global_index].y;
#ifdef FLOAT_PRECISION
	atomicAdd(&(spectrum[idx]), (DOUBLE) normmal);
	atomicAdd(&(count[idx]), (DOUBLE) 1.);
#else
	atomicAdd_double(&(spectrum[idx]), (DOUBLE) normmal);
	atomicAdd_double(&(count[idx]), (DOUBLE) 1.);
#endif

	}
	__syncthreads();
	if (threadIdx.x < ori_size)
	{
		spectrum[threadIdx.x] /= count[threadIdx.x];
		spectrum[threadIdx.x] *= (normfft / 2.);
	}
	if (threadIdx.x  < data_vs_prior_size)
	{
		tau2_D[threadIdx.x] =  tau2_fudge * spectrum[threadIdx.x];
	}
}

void update_tau2_gpu(CUFFT_COMPLEX * Fconv_D, DOUBLE* tau2_D, DOUBLE tau2_fudge, int xdim, int ydim, int zdim, int data_vs_prior_size, int ori_size,  DOUBLE  normfft)
{

	dim3 dimBlock((ori_size >= 512) ? ori_size : 512, 1, 1);
	dim3 dimGrid(1, 1, 1);
	int shared_mem_size = sizeof(DOUBLE) * ori_size * 2;
	update_tau2_kernel <<< dimGrid, dimBlock, shared_mem_size>>>(Fconv_D, tau2_D, tau2_fudge, xdim, ydim, zdim, data_vs_prior_size, ori_size, normfft);


}

void BackProjector::reconstruct_gpu(MultidimArray<DOUBLE>& vol_out,
                                    int max_iter_preweight,
                                    bool do_map,
                                    DOUBLE tau2_fudge,
                                    MultidimArray<DOUBLE>& tau2,
                                    MultidimArray<DOUBLE>& sigma2,
                                    MultidimArray<DOUBLE>& data_vs_prior,
                                    MultidimArray<DOUBLE> fsc, // only input
                                    DOUBLE normalise,
                                    bool update_tau2_with_fsc,
                                    bool is_whole_instead_of_half,
                                    int nr_threads,
                                    int minres_map)

{


	FourierTransformer transformer;

	//MultidimArray<Complex > Fconv;
	//MultidimArray<DOUBLE> Fweight, Fnewweight;
	int max_r2 = r_max * r_max * padding_factor * padding_factor;

	size_t free, total;
	cudaMemGetInfo(&free,  &total);
	std::cout << "GPU memor  y info total beginnig1 " << total / (1024 * 1024) << "MB  free  memory " << free / (1024 * 1024) << " MB "  << std::endl;
	// At the x=0 line, we have collected either the positive y-z coordinate, or its negative Friedel pair.
	// Sum these two together for both the data and the weight arrays
	CUFFT_COMPLEX * data_D;
	DOUBLE* weight_D;
	int xdim = data.xdim;
	int ydim = data.ydim;
	int xydim = data.yxdim;
	int zdim = data.zdim;
	int start_x = STARTINGX(data);
	int start_y = STARTINGY(data);
	int start_z = STARTINGZ(data);
	cudaMalloc((void**)&data_D, data.zyxdim * sizeof(CUFFT_COMPLEX ));
	cudaMalloc((void**)&weight_D, data.zyxdim * sizeof(DOUBLE));
	cudaMemcpy(data_D, data.data, data.zyxdim * sizeof(CUFFT_COMPLEX ), cudaMemcpyHostToDevice);
	cudaMemcpy(weight_D, weight.data, data.zyxdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	if (data.zdim > 1)
	{
		enforceHermitianSymmetry_gpu(data_D - start_x, weight_D - start_x, xdim, ydim, xydim, zdim);
	}

	cudaError cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel symmetrise_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}
	symmetrise_gpu(data_D,
	               weight_D,
	               xdim,
	               ydim,
	               xydim,
	               zdim,
	               start_x,
	               start_y,
	               start_z,
	               max_r2);
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel symmetrise_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}

	int new_xdim = pad_size / 2 + 1;
	int new_ydim = pad_size;
	int new_zdim = (ref_dim == 2) ? 1 : pad_size;
	int new_model_size = new_xdim * new_ydim * new_zdim;
	DOUBLE* Fweight_D;
       // Fnewweight can become too large for a float: always keep this one in double-precision
	double *Fnewweight_D;
	DOUBLE* sigma2_D,  *counter_D;
	CUFFT_COMPLEX * Fconv_D;
	DOUBLE* vol_out_D;

	/*if (ref_dim == 2)
	{
		cudaMalloc((void**)&Fconv_D, (pad_size / 2 + 1)*pad_size * sizeof(DOUBLE) * 2);
	}
	else
	{
		cudaMalloc((void**)&Fconv_D, (pad_size / 2 + 1)*pad_size * pad_size * sizeof(DOUBLE) * 2);
	}
*/
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel symmetrise_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}
	// clear vol_out to save memory!

	// Take oversampling into account
	DOUBLE oversampling_correction = (ref_dim == 3) ? (padding_factor * padding_factor * padding_factor) : (padding_factor * padding_factor);

	// First calculate the radial average of the (inverse of the) power of the noise in the reconstruction
	// This is the left-hand side term in the nominator of the Wiener-filter-like update formula
	// and it is stored inside the weight vector
	// Then, if (do_map) add the inverse of tau2-spectrum values to the weight
	cudaMalloc((void**) & sigma2_D, (ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMalloc((void**) & counter_D, (ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMemset(sigma2_D, 0., (ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMemset(counter_D, 0., (ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMalloc((void**) & Fweight_D, new_model_size * sizeof(DOUBLE));
	cudaMalloc((void**) & Fnewweight_D, new_model_size * sizeof(double));

	cudaMemset(Fweight_D, 0., new_model_size * sizeof(DOUBLE));
	cudaMemset(Fnewweight_D, 0., new_model_size * sizeof(double));

	decenter_gpu(weight_D,
	             Fweight_D,
	             max_r2,
	             new_xdim,
	             new_ydim,
	             new_zdim,
	             xdim,
	             ydim,
	             start_x,
	             start_y,
	             start_z);

	calculate_sigma2_gpu(
	    Fweight_D,
	    sigma2_D,
	    counter_D,
	    padding_factor,
	    oversampling_correction,
	    new_xdim,
	    new_ydim,
	    new_zdim, max_r2,
	    ori_size / 2 + 1);

	DOUBLE* fsc_D, *tau2_D, *data_vs_prior_D;
	cudaMalloc((void**) &fsc_D, fsc.xdim * sizeof(DOUBLE));
	cudaMalloc((void**) &tau2_D, tau2.xdim * sizeof(DOUBLE));
	cudaMalloc((void**) &data_vs_prior_D, (ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMemcpy(fsc_D, fsc.data, fsc.xdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cudaMemcpy(tau2_D, tau2.data, tau2.xdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cudaMemset(data_vs_prior_D, 0., (ori_size / 2 + 1)*sizeof(DOUBLE));

	//Due to the value of (ori_size/2 + 1) is very limited, we remain the follow section to be processed in CPU side
	if (update_tau2_with_fsc)
	{
		update_tau2_with_fsc_gpu(sigma2_D, fsc_D, tau2_D, data_vs_prior_D, (ori_size / 2 + 1), is_whole_instead_of_half);
	}

	if (do_map)
	{
		if (!update_tau2_with_fsc)
		{
			cudaMemset(data_vs_prior_D, 0., (ori_size / 2 + 1)*sizeof(DOUBLE));
		}
		cudaMemset(counter_D, 0., (ori_size / 2 + 1)*sizeof(DOUBLE));
		Applymap_additional_to_weight_gpu(Fweight_D, tau2_D, data_vs_prior_D, counter_D,
		                                  max_r2, new_xdim, new_ydim, new_zdim,
		                                  padding_factor, oversampling_correction, tau2_fudge, update_tau2_with_fsc, minres_map);
		if (!update_tau2_with_fsc)
		{
			Average_data_vs_prior_gpu(data_vs_prior_D, counter_D, ori_size / 2 + 1, r_max);
		}

	}

	// Divide both data and Fweight by normalisation factor to prevent FFT's with very large values....
	//std::cout <<"Running the GPU Fweight_D  after:" << data.zdim << std::endl;
	do_normalise_weight_data_gpu(Fweight_D, data_D, new_model_size, data.zyxdim, normalise);
	init_Fnewweight_gpu(Fnewweight_D, new_xdim, new_ydim, new_zdim, max_r2);


	int tab_size = tab_ftblob.tabulatedValues.xdim;
	DOUBLE* tabulatedValues_D;
	cudaMalloc((void**)&tabulatedValues_D, tab_size * sizeof(DOUBLE));
	cudaMemcpy(tabulatedValues_D, tab_ftblob.tabulatedValues.data, tab_size * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	// Iterative algorithm as in  Eq. [14] in Pipe & Menon (1999)
	// or Eq. (4) in Matej (2001)
	cufftHandle fPlanForward_gpu;
	cufftHandle fPlanBackward_gpu;
	cufftResult fftplan1, fftplan2;
//	size_t free, total;
	cudaMemGetInfo(&free,  &total);
	std::cout << "GPU memory info total 1 " << total / (1024 * 1024) << "MB  free  memory " << free / (1024 * 1024) << " MB "  << std::endl;
	std::cout << "cufft need memory " << pad_size * pad_size * (ref_dim == 2? 1: pad_size) << std::endl;

	//std::cout << "The fft plan size is " << pad_size* pad_size* (ref_dim == 2 ? 1 : pad_size) << " paded " << pad_size << std::endl;
#ifdef FLOAT_PRECISION
	fftplan1 = cufftPlan3d(&fPlanBackward_gpu ,  pad_size, pad_size, (ref_dim == 2 ? 1 : pad_size), CUFFT_C2R);
#else
	fftplan1 = cufftPlan3d(&fPlanBackward_gpu ,  pad_size, pad_size, (ref_dim == 2 ? 1 : pad_size), CUFFT_Z2D);
#endif
	
	if (fPlanBackward_gpu == NULL)
	{
		std::cerr << " fftplan create failed fPlanBackward_gpu= " << fftplan1 << " fPlanBackward= "   << " iter " << pad_size << std::endl;
	}
#ifdef FLOAT_PRECISION
		fftplan2 = cufftPlan3d(&fPlanForward_gpu ,  pad_size, pad_size, (ref_dim == 2 ? 1 : pad_size), CUFFT_R2C);
#else
		fftplan2 = cufftPlan3d(&fPlanForward_gpu ,  pad_size, pad_size, (ref_dim == 2 ? 1 : pad_size), CUFFT_D2Z);
#endif
	
	if (fPlanForward_gpu == NULL)
	{
		std::cerr << " fftplan create failed fPlanForward_gpu= " << fftplan2 << " fPlanForward_gpu= "    << " iter " << ref_dim << std::endl;
	}
	DOUBLE* Mconv_D;
	if (ref_dim == 2)
	{
		cudaMalloc((void**)&Mconv_D, pad_size * pad_size * sizeof(DOUBLE));
		cudaMalloc((void**)&Fconv_D, (pad_size / 2 + 1)*pad_size * sizeof(DOUBLE) * 2);
		cudaMemset(Mconv_D, 0., pad_size * pad_size * sizeof(DOUBLE));
		cudaMemset(Fconv_D, 0., (pad_size / 2 + 1)*pad_size * sizeof(DOUBLE) * 2);
	}
	else
	{
		cudaMalloc((void**)&Mconv_D, pad_size * pad_size * pad_size * sizeof(DOUBLE));
		cudaMalloc((void**)&Fconv_D, (pad_size / 2 + 1)*pad_size * pad_size * sizeof(DOUBLE) * 2);
		cudaMemset(Mconv_D, 0., pad_size * pad_size * pad_size * sizeof(DOUBLE));
		cudaMemset(Fconv_D, 0., (pad_size / 2 + 1)*pad_size * pad_size * sizeof(DOUBLE) * 2);
	}

	for (int iter = 0; iter < max_iter_preweight; iter++)
	{

		init_Fconv_gpu(Fconv_D, Fnewweight_D, Fweight_D, new_model_size);
		//======================================================
		DOUBLE normftblob = tab_ftblob(0.);
#ifdef FLOAT_PRECISION
		cufftExecC2R(fPlanBackward_gpu,  Fconv_D, Mconv_D);
#else
	       cufftExecZ2D(fPlanBackward_gpu,  Fconv_D, Mconv_D);
#endif

		//transformer.inverseFourierTransform_gpu(Fconv_D, Mconv_D, 1, pad_size, pad_size, (ref_dim==2?1:pad_size));
		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess)
		{
			printf("kernel symmetrise_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}
		Multi_by_FT_tab_gpu(Mconv_D, tabulatedValues_D,  tab_ftblob(0.),  tab_ftblob.sampling, pad_size, pad_size / 2, ori_size * padding_factor, padding_factor, tab_ftblob.tabulatedValues.xdim, false);
#ifdef FLOAT_PRECISION
		cufftExecR2C(fPlanForward_gpu,  Mconv_D, Fconv_D);
#else
	       cufftExecD2Z(fPlanForward_gpu,  Mconv_D, Fconv_D);
#endif

		ScaleComplexPointwise_gpu(Fconv_D, (pad_size / 2 + 1)*pad_size * (ref_dim == 2 ? 1 : pad_size), 1.0 / (pad_size * pad_size * (ref_dim == 2 ? 1 : pad_size)));

		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess)
		{
			printf("kernel symmetrise_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}
		update_Fconv_gpu(Fconv_D, Fnewweight_D, new_xdim, new_ydim, new_zdim,  max_r2);

	}
	cufftDestroy(fPlanForward_gpu);
	cufftDestroy(fPlanBackward_gpu);
	cudaMemset(Fconv_D, 0., new_model_size * sizeof(CUFFT_COMPLEX ));

	decenter_gpu(data_D,
	             Fconv_D,
	             Fnewweight_D,
	             max_r2,
	             new_xdim,
	             new_ydim,
	             new_zdim,
	             xdim,
	             ydim,
	             start_x,
	             start_y,
	             start_z);

	tau2.initZeros(ori_size / 2 + 1);
	data_vs_prior.initZeros(ori_size / 2 + 1);
	sigma2.initZeros(ori_size / 2 + 1);
	if (update_tau2_with_fsc)
	{
		cudaMemcpy(tau2.data, tau2_D, (ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	}
	cudaMemcpy(sigma2.data, sigma2_D, (ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	cudaMemcpy(data_vs_prior.data, data_vs_prior_D, (ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);

	cudaMemcpy(data.data, data_D, data.zyxdim * sizeof(CUFFT_COMPLEX ), cudaMemcpyDeviceToHost);
	cudaMemcpy(weight.data, weight_D, data.zyxdim * sizeof(DOUBLE), cudaMemcpyDeviceToHost);


	// Now do inverse FFT and window to original size in real-space
	// Pass the transformer to prevent making and clearing a new one before clearing the one declared above....
	// The latter may give memory problems as detected by electric fence....
	//windowToOridimRealSpace(transformer, Fconv, vol_out, nr_threads);
	int padoridim = padding_factor * ori_size;
	if (ref_dim == 2)
	{
		cudaMalloc((void**)&vol_out_D, padoridim * padoridim * sizeof(DOUBLE)); //Mout.resize(padoridim, padoridim);
		cudaMemset(vol_out_D, 0., padoridim * padoridim * sizeof(DOUBLE));

	}
	else
	{
		cudaMalloc((void**)&vol_out_D, padoridim * padoridim * padoridim * sizeof(DOUBLE));
		cudaMemset(vol_out_D, 0., padoridim * padoridim * padoridim * sizeof(DOUBLE));
	}
	//release some memory
	cudaFree(data_D);
	cudaFree(weight_D);
	cudaFree(Fweight_D);
	cudaFree(Fnewweight_D);
	cudaFree(tabulatedValues_D);
	cudaFree(Mconv_D);
	windowToOridimRealSpace_gpu(transformer, Fconv_D,
	                            vol_out_D,
	                            new_xdim,
	                            new_ydim,
	                            new_zdim);

	// Correct for the linear/nearest-neighbour interpolation that led to the data array
	griddingCorrect_gpu(vol_out_D,  ori_size, ori_size, ori_size,  interpolator,  r_min_nn,  ori_size * padding_factor);
	vol_out.resize((ref_dim == 2 ? 1 : ori_size), ori_size, ori_size);
	vol_out.setXmippOrigin();
	cudaMemcpy(vol_out.data, vol_out_D, (ori_size * ori_size * (ref_dim == 2 ? 1 : ori_size)*sizeof(DOUBLE)), cudaMemcpyDeviceToHost);

	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}
	
	cudaFree(Fconv_D);
	// If the tau-values were calculated based on the FSC, then now re-calculate the power spectrum of the actual reconstruction
	if (update_tau2_with_fsc)
	{

		DOUBLE* temp;
		cudaMalloc((void**)&temp, (ori_size * ori_size * (ref_dim == 2 ? 1 : ori_size)*sizeof(DOUBLE)));

		cudaMemcpy(temp, vol_out_D, (ori_size * ori_size * (ref_dim == 2 ? 1 : ori_size)*sizeof(DOUBLE)), cudaMemcpyDeviceToDevice);
		cudaMalloc((void**)&Fconv_D, ((ori_size / 2 + 1) * ori_size * (ref_dim == 2 ? 1 : ori_size)*sizeof(DOUBLE) * 2));
#ifdef FLOAT_PRECISION
		fftplan1 = cufftPlan3d(&fPlanForward_gpu ,  ori_size, ori_size, (ref_dim == 2 ? 1 : ori_size), CUFFT_R2C);
#else
		fftplan1 = cufftPlan3d(&fPlanForward_gpu ,  ori_size, ori_size, (ref_dim == 2 ? 1 : ori_size), CUFFT_D2Z);
#endif
		if (fPlanForward_gpu == NULL)
		{
			std::cerr << " fftplan create failed fPlanBackward_gpu= " << fftplan1 << " fPlanBackward= "   << std::endl;
			REPORT_ERROR("CUFFT Error: Unable to create plan");
		}
#ifdef FLOAT_PRECISION
		cufftExecR2C(fPlanForward_gpu,  temp, Fconv_D);
#else
		cufftExecD2Z(fPlanForward_gpu,  temp, Fconv_D);
#endif
		ScaleComplexPointwise_gpu(Fconv_D, (ori_size / 2 + 1)*ori_size * (ref_dim == 2 ? 1 : ori_size), 1.0 / (ori_size * ori_size * (ref_dim == 2 ? 1 : ori_size)));
		cufftDestroy(fPlanForward_gpu);

		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess)
		{
			printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}
		update_tau2_gpu(Fconv_D, tau2_D, tau2_fudge, (ori_size / 2 + 1) , ori_size, (ref_dim == 2 ? 1 : ori_size), (ori_size / 2 + 1) , ori_size, (ref_dim == 3) ? (DOUBLE)(ori_size * ori_size) : 1.);
		cudaMemcpy(tau2.data, tau2_D, (ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);
		cudaFree(Fconv_D);
		cudaFree(temp);

	}
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}
	// Completely empty the transformer object

	cudaFree(sigma2_D);
	cudaFree(counter_D);
	cudaFree(fsc_D);
	cudaFree(tau2_D);
	cudaFree(data_vs_prior_D);
	cudaFree(vol_out_D);

	//transformer.free_memory_gpu();
	transformer.cleanup();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}
	cudaMemGetInfo(&free,  &total);
	std::cout << "GPU memor  y info total end " << total / (1024 * 1024) << "MB  free  memory " << free / (1024 * 1024) << " MB "  << std::endl;
}

__global__ void enforceHermitianSymmetry_kernel(CUFFT_COMPLEX * my_data,
                                                DOUBLE* my_weight,
                                                int xdim,
                                                int ydim,
                                                int xydim,
                                                int zdim
                                               )
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	int yindex = global_index % ydim;
	int zindex = global_index / ydim;
	if (zindex > (zdim / 2))
	{
		return;
	}

	DOUBLE real1, real2, img1, img2;
	DOUBLE sum;
	if (zindex == (zdim / 2))
	{
		if (yindex < (ydim / 2))
		{
			real1 = my_data[zindex * xydim + yindex * xdim].x;
			img1 = my_data[zindex * xydim + yindex * xdim].y;
			real2 = my_data[((zdim - 1) - zindex) * xydim + (ydim - 1 - yindex) * xdim].x;
			img2 = my_data[((zdim - 1) - zindex) * xydim + (ydim - 1 - yindex) * xdim].y;

			my_data[zindex * xydim + yindex * xdim].x = real1 + real2;
			my_data[zindex * xydim + yindex * xdim].y = img1 - img2;
			my_data[((zdim - 1) - zindex)*xydim + (ydim - 1 - yindex)*xdim].x = real1 + real2;
			my_data[((zdim - 1) - zindex)*xydim + (ydim - 1 - yindex)*xdim].y = img2 - img1;

			sum = my_weight[zindex * xydim + yindex * xdim] + my_weight[(zdim - 1 - zindex) * xydim + (ydim - 1 - yindex) * xdim];
			my_weight[zindex * xydim + yindex * xdim] = sum;
			my_weight[((zdim - 1) - zindex)*xydim + (ydim - 1 - yindex)*xdim] = sum;
		}
	}
	else
	{
		real1 = my_data[zindex * xydim + yindex * xdim].x;
		img1 = my_data[zindex * xydim + yindex * xdim].y;
		real2 = my_data[((zdim - 1) - zindex) * xydim + (ydim - 1 - yindex) * xdim].x;
		img2 = my_data[((zdim - 1) - zindex) * xydim + (ydim - 1 - yindex) * xdim].y;

		my_data[zindex * xydim + yindex * xdim].x = real1 + real2;
		my_data[zindex * xydim + yindex * xdim].y = img1 - img2;
		my_data[((zdim - 1) - zindex)*xydim + (ydim - 1 - yindex)*xdim].x = real1 + real2;
		my_data[((zdim - 1) - zindex)*xydim + (ydim - 1 - yindex)*xdim].y = img2 - img1;

		sum = my_weight[zindex * xydim + yindex * xdim] + my_weight[((zdim - 1) - zindex) * xydim + (ydim - 1 - yindex) * xdim];
		my_weight[zindex * xydim + yindex * xdim] = sum;
		my_weight[((zdim - 1) - zindex)*xydim + (ydim - 1 - yindex)*xdim] = sum;
	}

}
void BackProjector::enforceHermitianSymmetry_gpu(CUFFT_COMPLEX * my_data_D,
                                                 DOUBLE* my_weight_D,
                                                 int xdim,
                                                 int ydim,
                                                 int xydim,
                                                 int zdim)
{
	int nr_pair_points = ydim * ((zdim + 1) / 2);
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((nr_pair_points + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);

	enforceHermitianSymmetry_kernel <<< gridDim, blockDim>>>(my_data_D,
	                                                         my_weight_D,
	                                                         xdim,
	                                                         ydim,
	                                                         xydim,
	                                                         zdim);

}

__global__ void symmetrise_kernel(const CUFFT_COMPLEX * __restrict__ my_data_temp_D ,
                                  const  DOUBLE* __restrict__ my_weight_temp_D,
                                  CUFFT_COMPLEX * my_data_D,
                                  DOUBLE* my_weight_D,
                                  int xdim,
                                  int ydim,
                                  int xydim,
                                  int zdim,
                                  int start_x,
                                  int start_y,
                                  int start_z,
                                  int my_rmax2,
                                  int nr_SymsNo)

{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	int x, y, z;
	x = global_index % xdim + start_x;
	y = (global_index / xdim) % ydim + start_y;
	z =  global_index / xydim + start_z;
	if ((x * x + y * y + z * z) > my_rmax2 || global_index >= xydim * zdim)
	{
		return;
	}

	DOUBLE  fx, fy, fz, xp, yp, zp;
	bool is_neg_x;
	int x0, x1, y0, y1, z0, z1;
	DOUBLE d000_r, d001_r, d010_r, d011_r, d100_r, d101_r, d110_r, d111_r;
	DOUBLE dx00_r, dx01_r, dx10_r, dx11_r, dxy0_r, dxy1_r;
	DOUBLE d000_i, d001_i, d010_i, d011_i, d100_i, d101_i, d110_i, d111_i;
	DOUBLE dx00_i, dx01_i, dx10_i, dx11_i, dxy0_i, dxy1_i;
	DOUBLE dd000, dd001, dd010, dd011, dd100, dd101, dd110, dd111;
	DOUBLE ddx00, ddx01, ddx10, ddx11, ddxy0, ddxy1;

	DOUBLE real, img, weight;
	weight = real = img = 0.;

	for (int i = 0; i < nr_SymsNo; i++)
	{
		// coords_output(x,y) = A * coords_input (xp,yp)

		xp = (DOUBLE)x * __R_array[i * 4 * 4] + (DOUBLE)y *  __R_array[i * 4 * 4 + 1] + (DOUBLE)z *  __R_array[i * 4 * 4 + 2];
		yp = (DOUBLE)x *  __R_array[i * 4 * 4 + 1 * 4] + (DOUBLE)y *  __R_array[i * 4 * 4 + 1 + 1 * 4] + (DOUBLE)z *  __R_array[i * 4 * 4 + 2 + 1 * 4];
		zp = (DOUBLE)x *  __R_array[i * 4 * 4 + 2 * 4] + (DOUBLE)y *  __R_array[i * 4 * 4 + 1 + 2 * 4] + (DOUBLE)z *  __R_array[i * 4 * 4 + 2 + 2 * 4];
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
		x0 = floor(xp);
		fx = xp - x0;
		x1 = x0 + 1;

		y0 = floor(yp);
		fy = yp - y0;
		y0 -=  start_y;
		y1 = y0 + 1;

		z0 = floor(zp);
		fz = zp - z0;
		z0 -= start_z;
		z1 = z0 + 1;

		// First interpolate (complex) data
		d000_r = my_data_temp_D[z0 * xydim + y0 * xdim + x0].x;
		d001_r = my_data_temp_D[z0 * xydim + y0 * xdim + x1].x;
		d010_r = my_data_temp_D[z0 * xydim + y1 * xdim + x0].x;
		d011_r = my_data_temp_D[z0 * xydim + y1 * xdim + x1].x;

		d000_i = my_data_temp_D[z0 * xydim + y0 * xdim + x0].y;
		d001_i = my_data_temp_D[z0 * xydim + y0 * xdim + x1].y;
		d010_i = my_data_temp_D[z0 * xydim + y1 * xdim + x0].y;
		d011_i = my_data_temp_D[z0 * xydim + y1 * xdim + x1].y;

		d100_r = my_data_temp_D[z1 * xydim + y0 * xdim + x0].x;
		d101_r = my_data_temp_D[z1 * xydim + y0 * xdim + x1].x;
		d110_r = my_data_temp_D[z1 * xydim + y1 * xdim + x0].x;
		d111_r = my_data_temp_D[z1 * xydim + y1 * xdim + x1].x;

		d100_i = my_data_temp_D[z1 * xydim + y0 * xdim + x0].y;
		d101_i = my_data_temp_D[z1 * xydim + y0 * xdim + x1].y;
		d110_i = my_data_temp_D[z1 * xydim + y1 * xdim + x0].y;
		d111_i = my_data_temp_D[z1 * xydim + y1 * xdim + x1].y;

		dx00_r = d000_r + (d001_r - d000_r) * fx;
		dx00_i = d000_i + (d001_i - d000_i) * fx;
		dx01_r = d100_r + (d101_r - d100_r) * fx;
		dx01_i = d100_i + (d101_i - d100_i) * fx;
		dx10_r = d010_r + (d011_r - d010_r) * fx;
		dx10_i = d010_i + (d011_i - d010_i) * fx;
		dx11_r = d110_r + (d111_r - d110_r) * fx;
		dx11_i = d110_i + (d111_i - d110_i) * fx;

		dxy0_r = dx00_r + (dx10_r - dx00_r) * fy;
		dxy0_i = dx00_i + (dx10_i - dx00_i) * fy;
		dxy1_r = dx01_r + (dx11_r - dx01_r) * fy;
		dxy1_i = dx01_i + (dx11_i - dx01_i) * fy;
		if (is_neg_x)
		{
			real += dxy0_r + (dxy1_r - dxy0_r) * fz;
			img -= (dxy0_i + (dxy1_i - dxy0_i) * fz);
		}
		else
		{
			real += dxy0_r + (dxy1_r - dxy0_r) * fz;
			img += (dxy0_i + (dxy1_i - dxy0_i) * fz);
		}

		// Then interpolate (real) weight
		dd000 = my_weight_temp_D[z0 * xydim + y0 * xdim + x0];
		dd001 = my_weight_temp_D[z0 * xydim + y0 * xdim + x1];
		dd010 = my_weight_temp_D[z0 * xydim + y1 * xdim + x0];
		dd011 = my_weight_temp_D[z0 * xydim + y1 * xdim + x1];
		dd100 = my_weight_temp_D[z1 * xydim + y0 * xdim + x0];
		dd101 = my_weight_temp_D[z1 * xydim + y0 * xdim + x1];
		dd110 = my_weight_temp_D[z1 * xydim + y1 * xdim + x0];
		dd111 = my_weight_temp_D[z1 * xydim + y1 * xdim + x1];

		ddx00 = dd000 + (dd001 - dd000) * fx;
		ddx01 = dd100 + (dd101 - dd100) * fx;
		ddx10 = dd010 + (dd011 - dd010) * fx;
		ddx11 = dd110 + (dd111 - dd110) * fx;
		ddxy0 = ddx00 + (ddx10 - ddx00) * fy;
		ddxy1 = ddx01 + (ddx11 - ddx01) * fy;
		weight += ddxy0 + (ddxy1 - ddxy0) * fz;

	}
	my_data_D[global_index].x += real;
	my_data_D[global_index].y += img;
	my_weight_D[global_index] += weight;
}
void BackProjector::symmetrise_gpu(CUFFT_COMPLEX * my_data_D,
                                   DOUBLE* my_weight_D,
                                   int xdim,
                                   int ydim,
                                   int xydim,
                                   int zdim,
                                   int start_x,
                                   int start_y,
                                   int start_z,
                                   int my_rmax2
                                  )
{
	if (SL.SymsNo() > 0 && ref_dim == 3)
	{
		int model_size = xydim * zdim;
		DOUBLE* my_weight_temp_D;
		CUFFT_COMPLEX * my_data_temp_D;
		cudaMemcpyToSymbol(__L_array, SL.__L.mdata, SL.SymsNo() * 4 * 4 * sizeof(DOUBLE), 0 , cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(__R_array, SL.__R.mdata, SL.SymsNo() * 4 * 4 * sizeof(DOUBLE), 0 , cudaMemcpyHostToDevice);
		dim3 blockDim(BLOCK_SIZE_128, 1, 1);
		dim3 gridDim((model_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);

		cudaMalloc((void**)&my_weight_temp_D, model_size * sizeof(DOUBLE));
		cudaMalloc((void**)&my_data_temp_D, model_size * sizeof(CUFFT_COMPLEX ));
		cudaMemcpy(my_data_temp_D, my_data_D, model_size * sizeof(CUFFT_COMPLEX ), cudaMemcpyDeviceToDevice);
		cudaMemcpy(my_weight_temp_D, my_weight_D, model_size * sizeof(DOUBLE), cudaMemcpyDeviceToDevice);
		cudaError cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess)
		{
			printf("kernel symmetrise_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}
		symmetrise_kernel <<< gridDim, blockDim >>>(my_data_temp_D,
		                                            my_weight_temp_D,
		                                            my_data_D,
		                                            my_weight_D,
		                                            xdim,
		                                            ydim,
		                                            xydim,
		                                            zdim,
		                                            start_x,
		                                            start_y,
		                                            start_z,
		                                            my_rmax2,
		                                            SL.SymsNo());
		cudaFree(my_data_temp_D);
		cudaFree(my_weight_temp_D);
	}

}

__global__ void decenter_kernel(const DOUBLE* __restrict__  weight_D, DOUBLE* Fweight_D, int max_r2,
                                int xdim, int ydim, int zdim, int xdim_weight, int ydim_weight,
                                int start_x, int start_y, int start_z)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	int i, j, k;
	int ip, jp, kp;
	j = global_index % xdim;
	i = (global_index / xdim) % ydim;
	k =  global_index / (xdim * ydim);

	jp = j;
	ip = (i < xdim) ? i : (i - ydim);
	kp = (k < xdim) ? k : (k - zdim);
	int ires = (kp * kp + ip * ip + jp * jp);
	if (global_index >= (xdim * ydim * zdim) || ires > max_r2)
	{
		return;
	}

	Fweight_D[global_index] = weight_D[(kp - start_z) * xdim_weight * ydim_weight + (ip - start_y) * xdim_weight + jp - start_x];

}

void BackProjector::decenter_gpu(DOUBLE* weight_D, DOUBLE* Fweight_D, int max_r2,
                                 int xdim, int ydim, int zdim, int xdim_weight, int ydim_weight,
                                 int start_x, int start_y, int start_z)
{
	int model_size = xdim * ydim * zdim;
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((model_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	decenter_kernel <<< gridDim, blockDim>>>(weight_D, Fweight_D, max_r2,
	                                         xdim, ydim, zdim, xdim_weight, ydim_weight,
	                                         start_x, start_y, start_z);

}

__global__ void decenter_kernel(const CUFFT_COMPLEX * __restrict__ data_D, CUFFT_COMPLEX * Fconv_D, const double* __restrict__ Fnewweight_D, int max_r2,
                                int xdim, int ydim, int zdim, int xdim_weight, int ydim_weight,
                                int start_x, int start_y, int start_z)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	int i, j, k;
	int ip, jp, kp;
	j = global_index % xdim;
	i = (global_index / xdim) % ydim;
	k =  global_index / (xdim * ydim);

	jp = j;
	ip = (i < xdim) ? i : (i - ydim);
	kp = (k < xdim) ? k : (k - zdim);
	int ires = (kp * kp + ip * ip + jp * jp);
	if (global_index >= (xdim * ydim * zdim) || ires > max_r2)
	{
		return;
	}
	double Fnewweight = Fnewweight_D[global_index];
#ifdef  FLOAT_PRECISION
            // Prevent numerical instabilities in single-precision reconstruction with very unevenly sampled orientations
            if (Fnewweight > 1e20)
                Fnewweight = 1e20;
#endif
	Fconv_D[global_index].x = data_D[(kp - start_z) * xdim_weight * ydim_weight + (ip - start_y) * xdim_weight + jp - start_x].x * Fnewweight;
	Fconv_D[global_index].y = data_D[(kp - start_z) * xdim_weight * ydim_weight + (ip - start_y) * xdim_weight + jp - start_x].y * Fnewweight;

}

void BackProjector::decenter_gpu(CUFFT_COMPLEX * data_D,
                                 CUFFT_COMPLEX * Fconv_D,
                                 double* Fnewweight_D,
                                 int max_r2,
                                 int xdim,
                                 int ydim,
                                 int zdim,
                                 int xdim_weight,
                                 int ydim_weight,
                                 int start_x,
                                 int start_y,
                                 int start_z)
{
	int model_size = xdim * ydim * zdim;
	dim3 blockDim(BLOCK_SIZE_128, 1, 1);
	dim3 gridDim((model_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, 1, 1);
	decenter_kernel <<< gridDim, blockDim>>>(data_D, Fconv_D, Fnewweight_D, max_r2,
	                                         xdim, ydim, zdim, xdim_weight, ydim_weight,
	                                         start_x, start_y, start_z);
}
void BackProjector::convoluteBlobRealSpace_gpu(FourierTransformer& transformer, DOUBLE* Mconv_D, DOUBLE* tabulatedValues_D, bool do_mask)
{

	// Blob normalisation in Fourier space
	DOUBLE normftblob = tab_ftblob(0.);

	transformer.setReal_gpu(Mconv_D, 1, pad_size, pad_size, (ref_dim == 2 ? 1 : pad_size));
	transformer.inverseTransform_gpu();

	Multi_by_FT_tab_gpu(Mconv_D, tabulatedValues_D,  tab_ftblob(0.),  tab_ftblob.sampling, pad_size, pad_size / 2, ori_size * padding_factor, padding_factor, tab_ftblob.tabulatedValues.xdim, false);


	transformer.Transform_gpu(1,  pad_size, pad_size, (ref_dim == 2) ? 1 : pad_size);

}
__global__ void windowFourierTransform_3D_kernel(CUFFT_COMPLEX * in, CUFFT_COMPLEX * out, int ixdim, int iydim, int izdim, int oxdim, int oydim, int ozdim)
{
	int index_within_img = threadIdx.x + blockIdx.x * blockDim.x;
	int i_offset, o_offset;

	i_offset = blockIdx.y * izdim * iydim * ixdim;
	o_offset = blockIdx.y * ozdim * oydim * oxdim;
	if (oxdim > ixdim)
	{
		long int max_r2 = (ixdim - 1) * (ixdim - 1);
		int i, j, k, ip, jp, kp;
		j = index_within_img % ixdim;
		i = (index_within_img / ixdim) % iydim;
		k =  index_within_img / (ixdim * iydim);
		jp = j;
		ip = (i < ixdim) ? i : (i - iydim);
		kp = (k < ixdim) ? k : (k - izdim);

		if (index_within_img >= (ixdim * iydim * izdim) || (kp * kp + ip * ip + jp * jp) > max_r2)
		{
			return;
		}
		int okp = (kp < 0) ? (kp + ozdim) : (kp);
		int oip = (ip < 0) ? (ip + oydim) : (ip);
		int ikp = (kp < 0) ? (kp + izdim) : (kp);
		int iip = (ip < 0) ? (ip + iydim) : (ip);
		out[okp * oydim * oxdim + oip * oxdim + jp + o_offset] =  in[ikp * iydim * ixdim + iip * ixdim + jp + i_offset];

	}
	else
	{
		int i, j, k, ip, jp, kp;
		j = index_within_img % oxdim;
		i = (index_within_img / oxdim) % oydim;
		k =  index_within_img / (oxdim * oydim);
		jp = j;
		ip = (i < oxdim) ? i : (i - oydim);
		kp = (k < oxdim) ? k : (k - ozdim);

		if (index_within_img >= (oxdim * oydim * ozdim))
		{
			return;
		}

		int ikp = (kp < 0) ? (kp + izdim) : (kp);
		int iip = (ip < 0) ? (ip + iydim) : (ip);

		out[index_within_img + o_offset] = in[ikp * iydim * ixdim + iip * ixdim + jp + i_offset];

	}

}
void windowFourierTransform_3D_gpu(CUFFT_COMPLEX * in,
                                   CUFFT_COMPLEX * out,
                                   int newdim,
                                   int nr_images,
                                   int ndim,
                                   int xdim,
                                   int ydim,
                                   int zdim)
{

	if (ydim > 1 && ydim / 2 + 1 != xdim)
	{
		REPORT_ERROR("windowFourierTransform ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");
	}
	long int newhdim = newdim / 2 + 1;

	if (newhdim == xdim)
	{
		cudaMemcpy(out, in, nr_images * zdim * ydim * xdim * sizeof(CUFFT_COMPLEX ), cudaMemcpyDeviceToDevice);
		return;
	}
	int out_size = newhdim * newdim * ((ndim == 2) ? 1 : newdim);
	dim3 dimBlock(BLOCK_SIZE_128, 1);
	dim3 dimGrid((out_size + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128, nr_images, 1);
	windowFourierTransform_3D_kernel <<< dimGrid, dimBlock>>>(in, out,
	                                                          xdim,
	                                                          (ndim > 1 ? ydim : 1),
	                                                          (ndim > 2 ? zdim : 1),
	                                                          newhdim,
	                                                          (ndim > 1 ? newdim : 1),
	                                                          (ndim > 2 ? newdim : 1));

}



void BackProjector::windowToOridimRealSpace_gpu(FourierTransformer& transformer,
                                                CUFFT_COMPLEX * Fin_D, DOUBLE* Mout_D,
                                                int new_xdim,
                                                int new_ydim,
                                                int new_zdim)
{
	int padoridim = padding_factor * ori_size;
	DOUBLE normfft;
	if (ref_dim == 2)
	{
		normfft = (DOUBLE)(padding_factor * padding_factor);
	}
	else
	{
		normfft = (DOUBLE)(padding_factor * padding_factor * padding_factor * ori_size);
	}


	CUFFT_COMPLEX * Ftmp_D;
	int fourier_size = (padoridim / 2 + 1) * padoridim * ((ref_dim == 2) ? 1 : padoridim);
	cudaMalloc((void**)&Ftmp_D, fourier_size * sizeof(CUFFT_COMPLEX ));
	cudaMemset(Ftmp_D, 0., fourier_size * sizeof(CUFFT_COMPLEX ));
	windowFourierTransform_3D_gpu(Fin_D,
	                              Ftmp_D,
	                              padoridim,
	                              1,
	                              ref_dim,
	                              new_xdim,
	                              new_ydim,
	                              new_zdim); // Do the inverse FFT
	cufftHandle fPlanBackward_gpu;
#ifdef FLOAT_PRECISION
	cufftResult fftplan1 = cufftPlan3d(&fPlanBackward_gpu ,  padoridim, padoridim, (ref_dim == 2 ? 1 : padoridim), CUFFT_C2R);
	cufftExecC2R(fPlanBackward_gpu,  Ftmp_D, Mout_D);	
#else
	cufftResult fftplan1 = cufftPlan3d(&fPlanBackward_gpu ,  padoridim, padoridim, (ref_dim == 2 ? 1 : padoridim), CUFFT_Z2D);
	cufftExecZ2D(fPlanBackward_gpu,  Ftmp_D, Mout_D);
#endif
	
	cufftDestroy(fPlanBackward_gpu);
	cudaFree(Ftmp_D);
	cudaError cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}

	DOUBLE* Mout_temp_D;
	if (ref_dim == 2)
	{
		cudaMalloc((void**)&Mout_temp_D, padoridim * padoridim * sizeof(DOUBLE)); //Mout.resize(padoridim, padoridim);
	}
	else
	{
		cudaMalloc((void**)&Mout_temp_D, padoridim * padoridim * padoridim * sizeof(DOUBLE));
	}

	centerFFT_2_gpu(Mout_D, Mout_temp_D, 1, ref_dim, padoridim, padoridim, ((ref_dim == 2) ? 1 : padoridim), true);

	window_gpu(Mout_temp_D, Mout_D, normfft, FIRST_XMIPP_INDEX(ori_size),  FIRST_XMIPP_INDEX(ori_size),  FIRST_XMIPP_INDEX(ori_size),
	           FIRST_XMIPP_INDEX(padoridim),  FIRST_XMIPP_INDEX(padoridim)
	           , FIRST_XMIPP_INDEX(((ref_dim == 2) ? 1 : padoridim)),
	           ori_size, ori_size, ((ref_dim == 2) ? 1 : ori_size), padoridim, padoridim, ((ref_dim == 2) ? 1 : padoridim));

	softMaskOutsideMap_new_gpu(Mout_D, -1., 3., (DOUBLE*) NULL, 1, ori_size, ori_size, ((ref_dim == 2) ? 1 : ori_size));
	cudaFree(Mout_temp_D);
}




