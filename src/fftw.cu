/***************************************************************************
 *
 * Author : "Huayou SU, Wen WEN, Xiaoli DU, Dongsheng LI"
 * Parallel and Distributed Processing Laboratory of NUDT
 * Author : "Maofu LIAO"
 * Department of Cell Biology, Harvard Medical School
 *
 * This file is GPU implementation of fftw in GeRelio.
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


#include "src/fftw.h"
#include "src/args.h"
#include <string.h>
#include <pthread.h>

#ifdef FLOAT_PRECISION
static pthread_mutex_t fftwf_plan_mutex = PTHREAD_MUTEX_INITIALIZER;
#else
static pthread_mutex_t fftw_plan_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif
static __device__  inline CUFFT_COMPLEX  ComplexScale(CUFFT_COMPLEX , DOUBLE);
static __global__ void ComplexPointwiseScale_kernel(CUFFT_COMPLEX *, int, DOUBLE);

template <typename T>
static __global__ void WindowOneImage_kernel(T*, T*, int , int , int , int , int , int);

static __global__ void selfApplyBeamTilt_kernel(CUFFT_COMPLEX *,
                                                DOUBLE*,
                                                DOUBLE*,
                                                DOUBLE*,
                                                DOUBLE*,
                                                DOUBLE ,
                                                int ,
                                                int);

template <typename T>
static __global__ void shiftImageInFourierTransform_1D_kernel(T*, T*,
                                                              DOUBLE*, int , int , int ,  int);
template <typename T>
static __global__ void shiftImageInFourierTransform_2D_kernel(T*, T*,
                                                              DOUBLE*, int , int , int ,  int, int);
template <typename T>
static __global__ void shiftImageInFourierTransform_3D_kernel(T*, T*,
                                                              DOUBLE*, int , int , int ,  int, int, int);

//========================================================================
//GPU  kernels' implementations
// Complex scalestatic
__device__  inline CUFFT_COMPLEX  ComplexScale(CUFFT_COMPLEX  a, DOUBLE s)
{
	CUFFT_COMPLEX  c;
	c.x = a.x * s;
	c.y = a.y * s;
	return c;
}
// Complex pointwise multiplication
static __global__ void ComplexPointwiseScale_kernel(CUFFT_COMPLEX * a, int size, DOUBLE scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads)
	{
		a[i] = ComplexScale(a[i], scale);
	}
}

template <typename T>
static __global__ void WindowOneImage_kernel(T* in, T* out, int ixdim, int iydim, int izdim, int oxdim, int oydim, int ozdim)
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int i_offset, o_offset;

	i_offset = blockIdx.x * izdim * iydim * ixdim;
	o_offset = blockIdx.x * ozdim * oydim * oxdim;
	if (oxdim > ixdim)
	{

		long int max_r2 = (ixdim - 1) * (ixdim - 1);
		int tid = tid_y * blockDim.x + tid_x;
		int image_size = izdim * iydim * ixdim;
		int block_size = blockDim.x * blockDim.y;
		int nr_loops = (image_size + block_size - 1) / block_size;
		for (int i = 0; i < nr_loops; i++)
		{
			int jp = tid % ixdim;
			int ip = (tid / ixdim);
			ip = (ip < ixdim) ? ip : (ip - iydim);
			int kp = tid / (iydim * ixdim);
			kp = (kp < ixdim) ? kp : (kp - izdim);
			if (kp * kp + ip * ip + jp * jp <= max_r2)
			{
				int okp = (kp < 0) ? (kp + ozdim) : (kp);
				int oip = (ip < 0) ? (ip + oydim) : (ip);
				int ikp = (kp < 0) ? (kp + izdim) : (kp);
				int iip = (ip < 0) ? (ip + iydim) : (ip);
				out[okp * oydim * oxdim + oip * oxdim + jp + o_offset] =  in[ikp * iydim * ixdim + iip * ixdim + jp + i_offset];
			}
			tid  += block_size;
		}
	}
	else
	{
		int tid = tid_y * blockDim.x + tid_x;
		int image_size = ozdim * oydim * oxdim;
		int block_size = blockDim.x * blockDim.y;
		int nr_loops = (image_size + block_size - 1) / block_size;
		for (int i = 0; i < nr_loops; i++)
		{
			if (tid < image_size)
			{
				int jp = tid % oxdim;
				int ip = tid / oxdim;
				ip = (ip >= oxdim) ? (ip - oydim) : ip;
				ip = (ip < 0) ? (ip + iydim) : ip;
				out[tid + o_offset] = in[jp + ip * ixdim + i_offset];
			}
			tid += block_size;
		}
	}

}

static __global__ void selfApplyBeamTilt_kernel(CUFFT_COMPLEX * Fimg_D,
                                                DOUBLE* beamtilt_x_D,
                                                DOUBLE* beamtilt_y_D,
                                                DOUBLE* wavelength_D,
                                                DOUBLE* Cs_D,
                                                DOUBLE boxsize,
                                                int xdim,
                                                int ydim)
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int offset;
	int tid = tid_y * blockDim.x + tid_x;
	int image_size = ydim * xdim;
	int block_size = blockDim.x * blockDim.y;
	int nr_loops = (image_size + block_size - 1) / block_size;
	offset = blockIdx.x * ydim * xdim;
	DOUBLE beamtilt_x, beamtilt_y, Cs, wavelength;
	beamtilt_x = beamtilt_x_D[blockIdx.x];
	beamtilt_y = beamtilt_y_D[blockIdx.x];
	Cs = Cs_D[blockIdx.x];
	wavelength = wavelength_D[blockIdx.x];
	if (beamtilt_x != 0 || beamtilt_y != 0)
	{
		DOUBLE factor = 0.360 * Cs * 10000000 * wavelength * wavelength / (boxsize * boxsize * boxsize);
		for (int i = 0; i < nr_loops; i++)
		{
			if (tid < image_size)
			{
				int jp = tid % xdim;
				int ip = (tid / xdim);
				ip = (ip < xdim) ? ip : (ip - ydim);

				DOUBLE delta_phase = factor * (ip * ip + jp * jp) * (ip * beamtilt_y + jp * beamtilt_x);
				DOUBLE realval = Fimg_D[tid + offset].x;
				DOUBLE imagval = Fimg_D[tid + offset].y;
				DOUBLE mag = sqrt(realval * realval + imagval * imagval);
				DOUBLE phas = atan2(imagval, realval) + DEG2RAD(delta_phase); // apply phase shift!
				realval = mag * cos(phas);
				imagval = mag * sin(phas);
				Fimg_D[tid + offset].x = realval;
				Fimg_D[tid + offset].y = imagval;
			}
			tid += block_size;
		}
	}

}
//==============================================================================
//C++ common functions
void FourierTransformer::setReal_gpu(CUFFT_REAL* V1,  int nr_images, int xdim, int ydim, int zdim)
{
	bool recomputePlan = false;
	if (dataPtr_D == NULL)
	{
		recomputePlan = true;
	}
	else if (dataPtr_D != V1)
	{
		recomputePlan = true;
	}
	long int mem_size = nr_images * zdim * ydim * (xdim / 2 + 1) * sizeof(CUFFT_COMPLEX );
	if (fFourier_D == NULL)
	{
		cudaMalloc((void**)&fFourier_D, mem_size);
		fourier_size = mem_size;
	}
	else if (fourier_size != mem_size)
	{
		fourier_size = mem_size ;
		if (fFourier_D != NULL)
		{
			cudaFree(fFourier_D);
		}
		cudaMalloc((void**)&fFourier_D, mem_size);
	}
	dataPtr_D = V1;
	if (recomputePlan)
	{
		int ndim = 3;
		if (zdim == 1)
		{
			ndim = 2;
			if (ydim == 1)
			{
				ndim = 1;
			}
		}
		int* N = new int[ndim];
		switch (ndim)
		{
		case 1:
			N[0] = xdim;
			break;
		case 2:
			N[0] = ydim;
			N[1] = xdim;
			break;
		case 3:
			N[0] = zdim;
			N[1] = ydim;
			N[2] = xdim;
			break;
		}

		// Destroy both forward and backward plans if they already exist
		destroyPlans_gpu();
		// Make new plans
#ifdef FLOAT_PRECISION
		cufftResult fftplan1 = cufftPlanMany(&fPlanForward_gpu, ndim, N,
		                                     NULL, 1, 0,
		                                     NULL, 1, 0,
		                                     CUFFT_R2C, nr_images);
#else
		cufftResult fftplan1 = cufftPlanMany(&fPlanForward_gpu, ndim, N,
		                                     NULL, 1, 0,
		                                     NULL, 1, 0,
		                                     CUFFT_D2Z, nr_images);
#endif
		
		if (fPlanForward_gpu == NULL)
		{
			std::cerr << " fftplan create failed fPlanForward= " << fftplan1 << " fPlanBackward= "   << std::endl;
			std::cerr << " fftplan create failed fPlanForward= " << xdim << " ydim= " << ydim << "zdim " << zdim << "fourzie " <<  fourier_size << std::endl;
			REPORT_ERROR("CUFFT Error: Unable to create plan");
		}
#ifdef FLOAT_PRECISION
		cufftResult fftplan2 =cufftPlanMany(&fPlanBackward_gpu, ndim, N,
		                                     NULL, 1, 0,
		                                     NULL, 1, 0,
		                                     CUFFT_C2R, nr_images);
#else
		cufftResult fftplan2 = cufftPlanMany(&fPlanBackward_gpu, ndim, N,
		                                     NULL, 1, 0,
		                                     NULL, 1, 0,
		                                     CUFFT_Z2D, nr_images);
#endif
		if (fPlanBackward_gpu == NULL)
		{
			std::cerr << " fftplan create failed fPlanBackward= " << fftplan2  << std::endl;
			std::cerr << " fftplan create failed fPlanBackward= " << xdim << " ydim= " << ydim << "zdim " << zdim << "fourzie " <<  fourier_size << std::endl;
			REPORT_ERROR("CUFFT Error: Unable to create plan");
		}

#ifdef DEBUG_PLANS
		std::cerr << " SETREAL fPlanForward= " << fPlanForward << " fPlanBackward= " << fPlanBackward  << " this= " << this << std::endl;
#endif

		delete [] N;

	}
}

void FourierTransformer::setReal_gpu(CUFFT_COMPLEX * V1, int nr_images, int xdim, int ydim, int zdim)
{
	bool recomputePlan = false;
	if (complexDataPtr_D != V1)
	{
		recomputePlan = true;
	}
	else if (complexDataPtr_D != V1)
	{
		recomputePlan = true;
	}
	int mem_size = nr_images * zdim * ydim * xdim * sizeof(CUFFT_COMPLEX );
	if (fFourier_D == NULL)
	{
		cudaMalloc((void**)&fFourier_D, mem_size);
		fourier_size = mem_size;
	}
	else if (fourier_size != mem_size)
	{
		fourier_size = mem_size ;
		if (fFourier_D != NULL)
		{
			cudaFree(fFourier_D);
		}
		cudaMalloc((void**)&fFourier_D, mem_size);
	}
	complexDataPtr_D = V1;
	if (recomputePlan)
	{
		int ndim = 3;
		if (zdim == 1)
		{
			ndim = 2;
			if (ydim == 1)
			{
				ndim = 1;
			}
		}
		int* N = new int[ndim];
		switch (ndim)
		{
		case 1:
			N[0] = xdim;
			break;
		case 2:
			N[0] = ydim;
			N[1] = xdim;
			break;
		case 3:
			N[0] = zdim;
			N[1] = ydim;
			N[2] = xdim;
			break;
		}


		//pthread_mutex_lock(&fftw_plan_mutex);
		if (fPlanForward_gpu != NULL)
		{
			cufftDestroy(fPlanForward_gpu);
		}
		fPlanForward_gpu = NULL;
#ifdef FLOAT_PRECISION
		cufftPlanMany(&fPlanForward_gpu, ndim, N,
		              NULL, 1, 0,
		              NULL, 1, 0,
		              CUFFT_Z2Z, nr_images);
#else
		cufftPlanMany(&fPlanForward_gpu, ndim, N,
		              NULL, 1, 0,
		              NULL, 1, 0,
		              CUFFT_C2C, nr_images);
#endif
		if (fPlanBackward_gpu != NULL)
		{
			cufftDestroy(fPlanBackward_gpu);
		}
		fPlanBackward_gpu = NULL;
#ifdef FLOAT_PRECISION
		cufftPlanMany(&fPlanBackward_gpu, ndim, N,
		              NULL, 1, 0,
		              NULL, 1, 0,
		              CUFFT_C2C, nr_images);
#else
		cufftPlanMany(&fPlanBackward_gpu, ndim, N,
		              NULL, 1, 0,
		              NULL, 1, 0,
		              CUFFT_Z2Z, nr_images);
#endif
		
		if (fPlanBackward_gpu == NULL || fPlanBackward_gpu == NULL)
		{
			REPORT_ERROR("CUFFT Error: Unable to create plan");
		}
		delete [] N;

		//pthread_mutex_unlock(&fftw_plan_mutex);
	}
}

void FourierTransformer::Transform_gpu(int nr_images, int xdim, int ydim, int zdim, int sign)
{
	if (sign == CUFFT_FORWARD)
	{
		// Normalisation of the transform
		int size = 0;
		size = zdim * ydim * (xdim / 2 + 1);
		if (dataPtr_D != NULL)
		{
#ifdef FLOAT_PRECISION
        cufftExecR2C(fPlanForward_gpu, (CUFFT_REAL*)dataPtr_D, (CUFFT_COMPLEX *) fFourier_D);
#else
        cufftExecD2Z(fPlanForward_gpu, (CUFFT_REAL*)dataPtr_D, (CUFFT_COMPLEX *) fFourier_D);
#endif
		//cufftExecD2Z(fPlanForward_gpu, (cufftDoubleReal*)dataPtr_D, (CUFFT_COMPLEX *) fFourier_D);

		}
		else if (complexDataPtr_D != NULL)
		{
#ifdef FLOAT_PRECISION
			cufftExecC2C(fPlanForward_gpu, (CUFFT_COMPLEX *)complexDataPtr_D, (CUFFT_COMPLEX *)fFourier_D, CUFFT_FORWARD);
#else
        		cufftExecZ2Z(fPlanForward_gpu, (CUFFT_COMPLEX *)complexDataPtr_D, (CUFFT_COMPLEX *)fFourier_D, CUFFT_FORWARD);
#endif
			//cufftExecZ2Z(fPlanForward_gpu, (CUFFT_COMPLEX *)complexDataPtr_D, (CUFFT_COMPLEX *)fFourier_D, CUFFT_FORWARD);
		}
		else
		{
			REPORT_ERROR("No complex nor real data defined");
		}
		cudaError cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess)
		{
			printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}

		dim3 blockDim(256, 1, 1);
		dim3 gridDim(nr_images, 1, 1);

		ComplexPointwiseScale_kernel <<< gridDim, blockDim>>>((CUFFT_COMPLEX *)fFourier_D, nr_images * size,  1.0 / (zdim * ydim * xdim));

		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess)
		{
			printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}
	}
}
void FourierTransformer::inverseTransform_gpu()
{
	if (dataPtr_D != NULL)
	{
#ifdef FLOAT_PRECISION
		cufftExecC2R(fPlanBackward_gpu, (CUFFT_COMPLEX *) fFourier_D, (CUFFT_REAL*)dataPtr_D);
#else
		cufftExecZ2D(fPlanBackward_gpu, (CUFFT_COMPLEX *) fFourier_D, (CUFFT_REAL*)dataPtr_D);
#endif
	}
	else if (complexDataPtr_D != NULL)
	{
#ifdef FLOAT_PRECISION
	cufftExecC2C(fPlanBackward_gpu, (CUFFT_COMPLEX *)fFourier_D, (CUFFT_COMPLEX *)complexDataPtr_D, CUFFT_INVERSE);
#else
	cufftExecZ2Z(fPlanBackward_gpu, (CUFFT_COMPLEX *)fFourier_D, (CUFFT_COMPLEX *)complexDataPtr_D, CUFFT_INVERSE);
#endif
	}
}
void FourierTransformer::destroyPlans_gpu()
{

	// Anything to do with plans has to be protected for threads!
	// pthread_mutex_lock(&fftw_plan_mutex);
	if (fPlanForward_gpu != NULL)
	{
#ifdef DEBUG_PLANS
		std::cerr << " DESTROY fPlanForward_gpu= " << fPlanForward_gpu  << " this= " << this << std::endl;
#endif
		//fftw_destroy_plan(fPlanForward);
		cufftDestroy(fPlanForward_gpu);

	}
	if (fPlanBackward_gpu != NULL)
	{
#ifdef DEBUG_PLANS
		std::cerr << " DESTROY fPlanBackward_gpu= " << fPlanBackward_gpu  << " this= " << this << std::endl;
#endif
		cufftDestroy(fPlanBackward_gpu);
	}

}

void FourierTransformer::setFourier_gpu(CUFFT_COMPLEX * inputFourier, int nr_images, int xdim, int ydim, int zdim)
{
	cudaMemcpy(fFourier_D, inputFourier,
	           nr_images * zdim * ydim * xdim * sizeof(CUFFT_COMPLEX ), cudaMemcpyDeviceToDevice);
}

void FourierTransformer::free_memory_gpu()
{
	cudaFree(fFourier_D);
}

template<typename T>
void windowFourierTransform_gpu(T* in,
                                T* out,
                                int newdim,
                                int nr_images,
                                int ndim,
                                int xdim,
                                int ydim,
                                int zdim)
{
	// Check size of the input array
	if (ydim > 1 && ydim / 2 + 1 != xdim)
	{
		REPORT_ERROR("windowFourierTransform ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");
	}
	long int newhdim = newdim / 2 + 1;
	// If same size, just return input


	if (newhdim == xdim)
	{
		cudaMemcpy(out, in, nr_images * zdim * ydim * xdim * sizeof(T), cudaMemcpyDeviceToDevice);
		return;
	}

	dim3 dimBlock(BLOCK_X, BLOCK_Y, 1);
	dim3 dimGrid(nr_images, 1, 1);

	WindowOneImage_kernel <<< dimGrid, dimBlock>>>(in, out,
	                                               xdim,
	                                               (ndim > 1 ? ydim : 1),
	                                               (ndim > 2 ? zdim : 1),
	                                               newhdim,
	                                               (ndim > 1 ? newdim : 1),
	                                               (ndim > 2 ? newdim : 1));
}

template void windowFourierTransform_gpu<CUFFT_COMPLEX >(CUFFT_COMPLEX * in,
                                                             CUFFT_COMPLEX * out,
                                                             int newdim,
                                                             int nr_images,
                                                             int ndim,
                                                             int xdim,
                                                             int ydim,
                                                             int zdim);
template void windowFourierTransform_gpu<DOUBLE>(DOUBLE* in,
                                                 DOUBLE* out,
                                                 int newdim,
                                                 int nr_images,
                                                 int ndim,
                                                 int xdim,
                                                 int ydim,
                                                 int zdim);

template<typename T>
void selfApplyBeamTilt_gpu(T* Fimg_D, DOUBLE* beamtilt_x_D, DOUBLE* beamtilt_y_D,
                           DOUBLE* wavelength, DOUBLE* Cs_D, DOUBLE angpix, int ori_size, int nr_images, int ndim, int xdim, int ydim)
{
	if (ndim != 2)
	{
		REPORT_ERROR("applyBeamTilt can only be done on 2D Fourier Transforms!");
	}
	else
	{
		dim3 dimBlock(BLOCK_X, BLOCK_Y, 1);
		dim3 dimGrid(nr_images, 1, 1);
		DOUBLE boxsize = angpix * ori_size;
		selfApplyBeamTilt_kernel <<< dimGrid, dimBlock>>>(Fimg_D,
		                                                  beamtilt_x_D,
		                                                  beamtilt_y_D,
		                                                  wavelength,
		                                                  Cs_D,
		                                                  boxsize,
		                                                  xdim,
		                                                  ydim);
	}


}

template void selfApplyBeamTilt_gpu<CUFFT_COMPLEX >(CUFFT_COMPLEX * Fimg_D, DOUBLE* beamtilt_x_D, DOUBLE* beamtilt_y_D,
                                                        DOUBLE* wavelength, DOUBLE* Cs_D, DOUBLE angpix, int ori_size, int nr_images, int ndim, int xdim, int ydim);


template<typename T>
static __global__ void shiftImageInFourierTransform_1D_kernel(T* in, T* out,
                                                              DOUBLE* shift, int nr_images, int nr_trans, int nr_oversampled_trans,  int xdim)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int trans_id = bid % (nr_trans * nr_oversampled_trans);
	DOUBLE xshift = shift[trans_id];
	T* local_in = in + bid * xdim;
	T* local_out = out + bid * xdim;
	if (abs(xshift) < XMIPP_EQUAL_ACCURACY)
	{
		 for (int j = tid; j < xdim; j+=blockDim.x)
	 	{
			 local_out[ j ]= local_in[ j ];
			//local_out[bid*xdim + j ].x= local_in[bid*xdim + j ].x;
	 		//local_out[bid*xdim + j ].y= local_in[bid*xdim + j ].y;
	 	}
	}
	int n_loop = (xdim + blockDim.x - 1) / blockDim.x;
	DOUBLE dotp, a, b, c, d, ac, bd, ab_cd, x;
	for (int i = 0 ; i < n_loop; i++)
	{
		x = tid + i * blockDim.x;
		if ((tid + i * blockDim.x) < xdim)
		{
			dotp = 2 * PI * (x * xshift);
			int idx = (int)(abs(dotp) / (2 * PI / 5000)) % 5000;

			a = cos(idx * 2 * PI / 5000);
			b = sin(idx * 2 * PI / 5000);
			b = (dotp > 0 ? b : (-b));
			c = local_in[tid + i * blockDim.x].x;
			d = local_in[tid + i * blockDim.x].y;
			ac = a * c;
			bd = b * d;
			ab_cd = (a + b) * (c + d); // (ab_cd-ac-bd = ad+bc : but needs 4 multiplications)
			local_out[ tid + i * blockDim.x].x = ac - bd;
			local_out[ tid + i * blockDim.x].y = ab_cd - ac - bd;
		}

	}

}
template<typename T>
static __global__ void shiftImageInFourierTransform_2D_kernel(T* in, T* out,
                                                              DOUBLE* shift, int nr_images, int nr_trans, int nr_oversampled_trans,  int xdim, int ydim, int _nr_elem)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int bid = blockIdx.x;
	int nr_loopx, nr_loopy;
	nr_loopx = (xdim + blockDim.x - 1) / blockDim.x;
	nr_loopy = (ydim + blockDim.y - 1) / blockDim.y;
	int tid = tidx+tidy*blockDim.x;
	int trans_id = bid % (nr_trans * nr_oversampled_trans);
	DOUBLE xshift = shift[trans_id * 2];
	DOUBLE yshift = shift[trans_id * 2 + 1];
	T* local_in = in + (bid / (nr_trans * nr_oversampled_trans)) * xdim * ydim;
	T* local_out = out + bid * xdim * ydim;
	if (abs(xshift) < XMIPP_EQUAL_ACCURACY && abs(yshift) < XMIPP_EQUAL_ACCURACY)
	{
		 for (int j = tid; j < xdim * ydim; j+=(blockDim.x*blockDim.y))
	 	{
	 		local_out[ j ]= local_in[ j ];
	 		//local_out[ j ].y= local_in[bid * xdim * ydim  + j ].y;
	 	}
	}
	DOUBLE dotp, a, b, c, d, ac, bd, ab_cd;
	DOUBLE sampling = 2 * 3.14159265358979323846 / (DOUBLE) _nr_elem;
	for (int j = 0; j < nr_loopy; j++)
	{
		DOUBLE y = (DOUBLE)tidy + j * blockDim.y;

		if ((tidy + j * blockDim.y) < ydim)
		{
			for (int i = 0; i < nr_loopx; i++)
			{
				DOUBLE x = (DOUBLE)tidx + i * blockDim.x;
				if ((tidx + i * blockDim.x) < xdim)
				{
					if (y >= (DOUBLE)xdim)
					{
						y = y - (DOUBLE)ydim;
					}
					dotp = 2 * 3.14159265358979323846 * (x * xshift + y * yshift);
					int idx = (int)(ABS(dotp) / sampling) % _nr_elem;
					DOUBLE xx = ((DOUBLE)idx * sampling);
					a = cos(xx);
					b = sin(xx);
					b = (dotp > 0 ? b : (-b));

					c = local_in[(tidy + j * blockDim.y) * xdim + i * blockDim.x + tidx].x;
					d = local_in[(tidy + j * blockDim.y) * xdim + i * blockDim.x + tidx].y;
					ac = a * c;
					bd = b * d;
					ab_cd = (a + b) * (c + d);
					local_out[(tidy + j * blockDim.y)*xdim + i * blockDim.x + tidx].x = ac - bd;
					local_out[(tidy + j * blockDim.y)*xdim + i * blockDim.x + tidx].y = ab_cd - ac - bd;
				}

			}
		}
	}
}

template<typename T>
static __global__ void shiftImageInFourierTransform_3D_kernel(T* in, T* out,
                                                              DOUBLE* shift, int nr_images, int nr_trans, int nr_oversampled_trans,  int xdim, int ydim, int zdim)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int bid = blockIdx.x;
	int tid = tidx + tidy * blockDim.x;
	int nr_loopx, nr_loopy;
	nr_loopx = (xdim + blockDim.x - 1) / blockDim.x;
	nr_loopy = (ydim + blockDim.y - 1) / blockDim.y;

	int trans_id = bid % (nr_trans * nr_oversampled_trans);
	DOUBLE xshift = shift[trans_id * 3];
	DOUBLE yshift = shift[trans_id * 3 + 1];
	DOUBLE zshift = shift[trans_id * 3 + 2];
	T* local_in = in + bid * xdim * ydim * zdim;
	T* local_out = out + bid * xdim * ydim * zdim;

	if (abs(xshift) < XMIPP_EQUAL_ACCURACY && abs(yshift) < XMIPP_EQUAL_ACCURACY && abs(zshift) < XMIPP_EQUAL_ACCURACY)
	{
		 for (int j = tid; j <  xdim * ydim * zdim; j+=blockDim.x*blockDim.y)
	 	{
			
	 		local_out[ j ]= local_in[ j ];
			//local_out[bid * xdim * ydim * zdim + j ].x= local_in[bid * xdim * ydim * zdim+ j ].x;
	 		//local_out[bid * xdim * ydim * zdim+ j ].y= local_in[bid * xdim * ydim * zdim + j ].y;
	 	}
	}

	DOUBLE dotp, a, b, c, d, ac, bd, ab_cd, x, y, z;
	for (long int k = 0; k < zdim; k++)
	{
		z = (k < xdim) ? k : (k - zdim);
		for (int j = 0; j < nr_loopy; j++)
		{
			y = (tidy + j * blockDim.y < xdim) ? (tidy + j * blockDim.y) : (tidy + j * blockDim.y - xdim);
			if (tidy + j * blockDim.y < ydim)
			{
				for (int i = 0 ; i < nr_loopx; i++)
				{
					x = tidx + i * blockDim.x;
					if (tidx + i * blockDim.x < xdim)
					{

						dotp = 2 * PI * (x * xshift + y * yshift + z * zshift);
						a = cos(dotp);
						b = sin(dotp);

						c = local_in[ tid + j * blockDim.y * xdim + i * blockDim.x + bid * xdim * ydim * zdim].x;
						d = local_in[ tid + j * blockDim.y * xdim + i * blockDim.x + bid * xdim * ydim * zdim].y;
						ac = a * c;
						bd = b * d;
						ab_cd = (a + b) * (c + d);
						local_out[ tid + j * blockDim.y * xdim + i * blockDim.x + bid * xdim * ydim * zdim].x = ac - bd;
						local_out[ tid + j * blockDim.y * xdim + i * blockDim.x + bid * xdim * ydim * zdim].y = ab_cd - ac - bd;
					}

				}
			}

		}
	}
}


template<typename T>
void shiftImageInFourierTransform_gpu(T* in,
                                      T* out,
                                      DOUBLE oridim, DOUBLE* shift,
                                      int nr_images,  int nr_trans, int nr_oversampled_trans,
                                      int xdim, int ydim, int zdim)
{

	int ndim = (zdim > 1 ? 3 : (ydim > 1 ? 2 : 1));
	int num_shift_imgs = nr_images * nr_trans * nr_oversampled_trans;
	int _nr_elem = 5000;
	DOUBLE* shift_D;
	DOUBLE* shift_host;
	shift_host = (DOUBLE*)malloc(sizeof(DOUBLE) * nr_trans * nr_oversampled_trans * ndim);
	cudaMalloc((void**)&shift_D, sizeof(DOUBLE)*nr_trans * nr_oversampled_trans * ndim);
	if (ndim == 1)
	{
		for (int i = 0 ; i < nr_trans * nr_oversampled_trans; i++)
		{
			shift_host[i] = (shift[i] / (-oridim));
		}
		cudaMemcpy(shift_D, shift_host, sizeof(DOUBLE)*nr_trans * nr_oversampled_trans, cudaMemcpyHostToDevice);
		dim3 blockDim(BLOCK_SIZE, 1, 1);
		dim3 gridDim(num_shift_imgs, 1, 1);
		shiftImageInFourierTransform_1D_kernel <<< gridDim, blockDim>>>(in, out, shift_D, nr_images,  nr_trans, nr_oversampled_trans, xdim);
	}
	else if (ndim == 2)
	{
		for (int i = 0 ; i < nr_trans * nr_oversampled_trans * ndim; i++)
		{
			shift_host[i] = (shift[i] / (-oridim));
		}
		cudaMemcpy(shift_D, shift_host, sizeof(DOUBLE)*nr_trans * nr_oversampled_trans * ndim, cudaMemcpyHostToDevice);
		dim3 blockDim(BLOCK_X, BLOCK_Y, 1);
		dim3 gridDim(num_shift_imgs, 1, 1);
		shiftImageInFourierTransform_2D_kernel <<< gridDim, blockDim>>>(in, out, shift_D, nr_images,  nr_trans, nr_oversampled_trans, xdim, ydim, _nr_elem);
	}
	else if (ndim == 3)
	{
		for (int i = 0 ; i < nr_trans * nr_oversampled_trans * ndim; i++)
		{
			shift_host[i] = (shift[i] / (-oridim));
		}
		cudaMemcpy(shift_D, shift_host, sizeof(DOUBLE)*nr_trans * nr_oversampled_trans * ndim, cudaMemcpyHostToDevice);
		dim3 blockDim(BLOCK_X, BLOCK_Y, 1);
		dim3 gridDim(num_shift_imgs, 1, 1);
		shiftImageInFourierTransform_3D_kernel <<< gridDim, blockDim>>>(in, out, shift_D, nr_images,  nr_trans, nr_oversampled_trans, xdim, ydim, zdim);
	}
	else
	{
		REPORT_ERROR("shiftImageInFourierTransform ERROR: dimension should be 1, 2 or 3!");
	}
	cudaFree(shift_D);
	free(shift_host);
}

template void shiftImageInFourierTransform_gpu<CUFFT_COMPLEX >(CUFFT_COMPLEX * in,
                                                                   CUFFT_COMPLEX * out,
                                                                   DOUBLE oridim, DOUBLE* shift,
                                                                   int nr_images,  int nr_trans, int nr_oversampled_trans,
                                                                   int xdim, int ydim, int zdim);
 template<typename T>
 static __global__ void shiftImageInFourierTransform_1D_kernel(T* in, T* out,
															   DOUBLE* shift,  bool* shift_flag_D, int nr_images,  int xdim)
 {
	 int tid = threadIdx.x;
	 int bid = blockIdx.x;
	 int trans_id = bid ;
	 DOUBLE xshift = shift[trans_id];
	 T* local_in = in + bid * xdim;
	 T* local_out = out + bid * xdim;
	 int n_loop = (xdim + blockDim.x - 1) / blockDim.x;
	 if (abs(xshift) < XMIPP_EQUAL_ACCURACY || !shift_flag_D[bid])
	 {
		   for (int j = tid; j < xdim; j+=blockDim.x)
	 	{
	 		
	 		local_out[ j ]= local_in[ j ];
	 		//local_out[bid*xdim + j ].x= local_in[bid*xdim + j ].x;
	 		//local_out[bid*xdim + j ].y= local_in[bid*xdim + j ].y;
	 	}
	 }
	
	 DOUBLE dotp, a, b, c, d, ac, bd, ab_cd, x;
	 for (int i = 0 ; i < n_loop; i++)
	 {
		 x = tid + i * blockDim.x;
		 if ((tid + i * blockDim.x) < xdim)
		 {
			 dotp = 2 * PI * (x * xshift);
			 int idx = (int)(abs(dotp) / (2 * PI / 5000)) % 5000;
 
			 a = cos(idx * 2 * PI / 5000);
			 b = sin(idx * 2 * PI / 5000);
			 b = (dotp > 0 ? b : (-b));
			 c = local_in[tid + i * blockDim.x].x;
			 d = local_in[tid + i * blockDim.x].y;
			 ac = a * c;
			 bd = b * d;
			 ab_cd = (a + b) * (c + d); // (ab_cd-ac-bd = ad+bc : but needs 4 multiplications)
			 local_out[ tid + i * blockDim.x].x = ac - bd;
			 local_out[ tid + i * blockDim.x].y = ab_cd - ac - bd;
		 }
 
	 }
 
 }
 template<typename T>
 static __global__ void shiftImageInFourierTransform_2D_kernel(T* in, T* out,
															   DOUBLE* shift, bool *shift_flag_D, 
															   int nr_images,  int xdim, int ydim, int _nr_elem)
 {
	 int tidx = threadIdx.x;
	 int tidy = threadIdx.y;
	 int bid = blockIdx.x;
	 int nr_loopx, nr_loopy, nr_loop;
	 nr_loopx = (xdim + blockDim.x - 1) / blockDim.x;
	 nr_loopy = (ydim + blockDim.y - 1) / blockDim.y;
 	 
	 int tid = tidx + tidy * blockDim.x;
	 int block_size =  blockDim.y * blockDim.x;
	 nr_loop = (xdim*ydim)/block_size +1;
	 int trans_id = bid ;
	 DOUBLE xshift = shift[trans_id * 2];
	 DOUBLE yshift = shift[trans_id * 2 + 1];
	 T* local_in = in + (bid ) * xdim * ydim;
	 T* local_out = out + bid * xdim * ydim;
	 if (abs(xshift) < XMIPP_EQUAL_ACCURACY && abs(yshift) < XMIPP_EQUAL_ACCURACY ||!shift_flag_D[bid] )
	 {
		  for (int j = tid; j < xdim*ydim; j+=block_size)
	 	{
	 		local_out[ j ]= local_in[ j ];
	 		//local_out[bid*xdim*ydim + j ].x= local_in[bid*xdim*ydim + j ].x;
			
	 		//local_out[bid*xdim*ydim + j ].y= local_in[bid*xdim*ydim + j ].y;
		 
	 	}
	 }
	 DOUBLE dotp, a, b, c, d, ac, bd, ab_cd;
	 DOUBLE sampling = 2 * 3.14159265358979323846 / (DOUBLE) _nr_elem;
	 for (int j = 0; j < nr_loopy; j++)
	 {
		 DOUBLE y = (DOUBLE)tidy + j * blockDim.y;
 
		 if ((tidy + j * blockDim.y) < ydim)
		 {
			 for (int i = 0; i < nr_loopx; i++)
			 {
				 DOUBLE x = (DOUBLE)tidx + i * blockDim.x;
				 if ((tidx + i * blockDim.x) < xdim)
				 {
					 if (y >= (DOUBLE)xdim)
					 {
						 y = y - (DOUBLE)ydim;
					 }
					 dotp = 2 * 3.14159265358979323846 * (x * xshift + y * yshift);
					 int idx = (int)(ABS(dotp) / sampling) % _nr_elem;
					 DOUBLE xx = ((DOUBLE)idx * sampling);
					 a = cos(dotp);
					 b = sin(dotp);
					 //b = (dotp > 0 ? b : (-b));
 
					 c = local_in[(tidy + j * blockDim.y) * xdim + i * blockDim.x + tidx].x;
					 d = local_in[(tidy + j * blockDim.y) * xdim + i * blockDim.x + tidx].y;
					 ac = a * c;
					 bd = b * d;
					 ab_cd = (a + b) * (c + d);
					 local_out[(tidy + j * blockDim.y)*xdim + i * blockDim.x + tidx].x = ac - bd;
					 local_out[(tidy + j * blockDim.y)*xdim + i * blockDim.x + tidx].y = ab_cd - ac - bd;
				 }
 
			 }
		 }
	 }
 }
 
 template<typename T>
 static __global__ void shiftImageInFourierTransform_3D_kernel(T* in, T* out,
															   DOUBLE* shift, 
															   bool *shift_flag_D, int nr_images,  int xdim, int ydim, int zdim)
 {
	 int tidx = threadIdx.x;
	 int tidy = threadIdx.y;
	 int bid = blockIdx.x;
	 int tid = tidx + tidy * blockDim.x;
	 int nr_loopx, nr_loopy;
	 nr_loopx = (xdim + blockDim.x - 1) / blockDim.x;
	 nr_loopy = (ydim + blockDim.y - 1) / blockDim.y;
 
	 int trans_id = bid ;
	 DOUBLE xshift = shift[trans_id * 3];
	 DOUBLE yshift = shift[trans_id * 3 + 1];
	 DOUBLE zshift = shift[trans_id * 3 + 2];
	 T* local_in = in + bid * xdim * ydim * zdim;
	 T* local_out = out + bid * xdim * ydim * zdim;
 
	 if (abs(xshift) < XMIPP_EQUAL_ACCURACY && abs(yshift) < XMIPP_EQUAL_ACCURACY && abs(zshift) < XMIPP_EQUAL_ACCURACY || !shift_flag_D[bid])
	 {
	 	 for (int j = tid; j <  xdim * ydim * zdim; j+=blockDim.x*blockDim.y)
	 	{
	 		
	 		local_out[ j ]= local_in[ j ];
	 		//local_out[bid * xdim * ydim * zdim + j ].x= local_in[bid * xdim * ydim * zdim+ j ].x;
	 		//local_out[bid * xdim * ydim * zdim+ j ].y= local_in[bid * xdim * ydim * zdim + j ].y;
	 	}

	 }
 
	 DOUBLE dotp, a, b, c, d, ac, bd, ab_cd, x, y, z;
	 for (long int k = 0; k < zdim; k++)
	 {
		 z = (k < xdim) ? k : (k - zdim);
		 for (int j = 0; j < nr_loopy; j++)
		 {
			 y = (tidy + j * blockDim.y < xdim) ? (tidy + j * blockDim.y) : (tidy + j * blockDim.y - xdim);
			 if (tidy + j * blockDim.y < ydim)
			 {
				 for (int i = 0 ; i < nr_loopx; i++)
				 {
					 x = tidx + i * blockDim.x;
					 if (tidx + i * blockDim.x < xdim)
					 {
 
						 dotp = 2 * PI * (x * xshift + y * yshift + z * zshift);
						 a = cos(dotp);
						 b = sin(dotp);
 
						 c = local_in[ tid + j * blockDim.y * xdim + i * blockDim.x + bid * xdim * ydim * zdim].x;
						 d = local_in[ tid + j * blockDim.y * xdim + i * blockDim.x + bid * xdim * ydim * zdim].y;
						 ac = a * c;
						 bd = b * d;
						 ab_cd = (a + b) * (c + d);
						 local_out[ tid + j * blockDim.y * xdim + i * blockDim.x + bid * xdim * ydim * zdim].x = ac - bd;
						 local_out[ tid + j * blockDim.y * xdim + i * blockDim.x + bid * xdim * ydim * zdim].y = ab_cd - ac - bd;
					 }
 
				 }
			 }
 
		 }
	 }
 }
 


 template<typename T>
 void shiftImageInFourierTransform_gpu(T* in,
									   T* out,
									   DOUBLE oridim, DOUBLE* shift,
									   bool* shift_flag_D, int nr_images, 
									   int xdim, int ydim, int zdim)
 {
 
	 int ndim = (zdim > 1 ? 3 : (ydim > 1 ? 2 : 1));
	 int num_shift_imgs = nr_images;
	 int _nr_elem = 5000;
	 DOUBLE* shift_D;
	 DOUBLE* shift_host;
	 shift_host = (DOUBLE*)malloc(sizeof(DOUBLE) *nr_images * ndim);
	 cudaMalloc((void**)&shift_D, sizeof(DOUBLE)*nr_images * ndim);
	 if (ndim == 1)
	 {
		 for (int i = 0 ; i < nr_images; i++)
		 {
			 shift_host[i] = (shift[i] / (-oridim));
		 }
		 cudaMemcpy(shift_D, shift_host, sizeof(DOUBLE)*nr_images, cudaMemcpyHostToDevice);
		 dim3 blockDim(BLOCK_SIZE, 1, 1);
		 dim3 gridDim(num_shift_imgs, 1, 1);
		 shiftImageInFourierTransform_1D_kernel <<< gridDim, blockDim>>>(in, out, shift_D, shift_flag_D, nr_images,  xdim);
	 }
	 else if (ndim == 2)
	 {
 
 
		 for (int i = 0 ; i < nr_images * ndim; i++)
		 {
			 shift_host[i] = (shift[i] / (-oridim));
		 }
		 cudaMemcpy(shift_D, shift_host, sizeof(DOUBLE)*nr_images* ndim, cudaMemcpyHostToDevice);
		 dim3 blockDim(BLOCK_X, BLOCK_Y, 1);
		 dim3 gridDim(num_shift_imgs, 1, 1);
		 shiftImageInFourierTransform_2D_kernel <<< gridDim, blockDim>>>(in, out, shift_D, shift_flag_D, nr_images, xdim, ydim, _nr_elem);
	 }
	 else if (ndim == 3)
	 {
		 for (int i = 0 ; i < nr_images * ndim; i++)
		 {
			 shift_host[i] = (shift[i] / (-oridim));
		 }
		 cudaMemcpy(shift_D, shift_host, sizeof(DOUBLE)*nr_images * ndim, cudaMemcpyHostToDevice);
		 dim3 blockDim(BLOCK_X, BLOCK_Y, 1);
		 dim3 gridDim(num_shift_imgs, 1, 1);
		 shiftImageInFourierTransform_3D_kernel <<< gridDim, blockDim>>>(in, out, shift_D, shift_flag_D, nr_images,  xdim, ydim, zdim);
	 }
	 else
	 {
		 REPORT_ERROR("shiftImageInFourierTransform ERROR: dimension should be 1, 2 or 3!");
	 }
	 cudaFree(shift_D);
	 free(shift_host);
 }

 template void shiftImageInFourierTransform_gpu<CUFFT_COMPLEX >(CUFFT_COMPLEX * in,
 															CUFFT_COMPLEX * out,
 															DOUBLE oridim, DOUBLE* shift,
 															bool * shift_flag_D,
 															int nr_images, 
 															int xdim, int ydim, int zdim);

