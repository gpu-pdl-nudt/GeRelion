/***************************************************************************
 *
 * Author: "Huayou SU"
 * PDL of NUDT
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

static pthread_mutex_t fftw_plan_mutex = PTHREAD_MUTEX_INITIALIZER;

static __device__  inline cufftDoubleComplex ComplexScale(cufftDoubleComplex, double);
static __global__ void ComplexPointwiseScale_kernel(cufftDoubleComplex*, int, double);

template <typename T>
static __global__ void WindowOneImage_kernel(T*, T*, int , int , int , int , int , int);

static __global__ void selfApplyBeamTilt_kernel(cufftDoubleComplex*,
                                                double*,
                                                double*,
                                                double*,
                                                double*,
                                                double ,
                                                int ,
                                                int);

template <typename T>
static __global__ void shiftImageInFourierTransform_1D_kernel(T*, T*,
                                                              double*, int , int , int ,  int);
template <typename T>
static __global__ void shiftImageInFourierTransform_2D_kernel(T*, T*,
                                                              double*, int , int , int ,  int, int);
template <typename T>
static __global__ void shiftImageInFourierTransform_3D_kernel(T*, T*,
                                                              double*, int , int , int ,  int, int, int);

//========================================================================
//GPU  kernels' implementations
// Complex scalestatic
__device__  inline cufftDoubleComplex ComplexScale(cufftDoubleComplex a, double s)
{
	cufftDoubleComplex c;
	c.x = a.x * s;
	c.y = a.y * s;
	return c;
}
// Complex pointwise multiplication
static __global__ void ComplexPointwiseScale_kernel(cufftDoubleComplex* a, int size, double scale)
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

static __global__ void selfApplyBeamTilt_kernel(cufftDoubleComplex* Fimg_D,
                                                double* beamtilt_x_D,
                                                double* beamtilt_y_D,
                                                double* wavelength_D,
                                                double* Cs_D,
                                                double boxsize,
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
	double beamtilt_x, beamtilt_y, Cs, wavelength;
	beamtilt_x = beamtilt_x_D[blockIdx.x];
	beamtilt_y = beamtilt_y_D[blockIdx.x];
	Cs = Cs_D[blockIdx.x];
	wavelength = wavelength_D[blockIdx.x];
	if (beamtilt_x != 0 || beamtilt_y != 0)
	{
		double factor = 0.360 * Cs * 10000000 * wavelength * wavelength / (boxsize * boxsize * boxsize);
		for (int i = 0; i < nr_loops; i++)
		{
			if (tid < image_size)
			{
				int jp = tid % xdim;
				int ip = (tid / xdim);
				ip = (ip < xdim) ? ip : (ip - ydim);

				double delta_phase = factor * (ip * ip + jp * jp) * (ip * beamtilt_y + jp * beamtilt_x);
				double realval = Fimg_D[tid + offset].x;
				double imagval = Fimg_D[tid + offset].y;
				double mag = sqrt(realval * realval + imagval * imagval);
				double phas = atan2(imagval, realval) + DEG2RAD(delta_phase); // apply phase shift!
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
void FourierTransformer::setReal_gpu(cufftDoubleReal* V1,  int nr_images, int xdim, int ydim, int zdim)
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
	long int mem_size = nr_images * zdim * ydim * (xdim / 2 + 1) * sizeof(cufftDoubleComplex);
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
		cufftResult fftplan1 = cufftPlanMany(&fPlanForward_gpu, ndim, N,
		                                     NULL, 1, 0,
		                                     NULL, 1, 0,
		                                     CUFFT_D2Z, nr_images);
		if (fPlanForward_gpu == NULL)
		{
			std::cerr << " fftplan create failed fPlanForward= " << fftplan1 << " fPlanBackward= "   << std::endl;
			std::cerr << " fftplan create failed fPlanForward= " << xdim << " ydim= " << ydim << "zdim " << zdim << "fourzie " <<  fourier_size << std::endl;
			REPORT_ERROR("CUFFT Error: Unable to create plan");
		}
		cufftResult fftplan2 = cufftPlanMany(&fPlanBackward_gpu, ndim, N,
		                                     NULL, 1, 0,
		                                     NULL, 1, 0,
		                                     CUFFT_Z2D, nr_images);

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

void FourierTransformer::setReal_gpu(cufftDoubleComplex* V1, int nr_images, int xdim, int ydim, int zdim)
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
	int mem_size = nr_images * zdim * ydim * xdim * sizeof(cufftDoubleComplex);
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


		pthread_mutex_lock(&fftw_plan_mutex);
		if (fPlanForward_gpu != NULL)
		{
			cufftDestroy(fPlanForward_gpu);
		}
		fPlanForward_gpu = NULL;
		cufftPlanMany(&fPlanForward_gpu, ndim, N,
		              NULL, 1, 0,
		              NULL, 1, 0,
		              CUFFT_Z2Z, nr_images);
		if (fPlanBackward_gpu != NULL)
		{
			cufftDestroy(fPlanBackward_gpu);
		}
		fPlanBackward_gpu = NULL;
		cufftPlanMany(&fPlanBackward_gpu, ndim, N,
		              NULL, 1, 0,
		              NULL, 1, 0,
		              CUFFT_Z2Z, nr_images);
		if (fPlanBackward_gpu == NULL || fPlanBackward_gpu == NULL)
		{
			REPORT_ERROR("CUFFT Error: Unable to create plan");
		}
		delete [] N;

		pthread_mutex_unlock(&fftw_plan_mutex);
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

			cufftExecD2Z(fPlanForward_gpu, (cufftDoubleReal*)dataPtr_D, (cufftDoubleComplex*) fFourier_D);

		}
		else if (complexDataPtr_D != NULL)
		{
			cufftExecZ2Z(fPlanForward_gpu, (cufftDoubleComplex*)complexDataPtr_D, (cufftDoubleComplex*)fFourier_D, CUFFT_FORWARD);
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

		ComplexPointwiseScale_kernel <<< gridDim, blockDim>>>((cufftDoubleComplex*)fFourier_D, nr_images * size,  1.0 / (zdim * ydim * xdim));

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
		cufftExecZ2D(fPlanBackward_gpu, (cufftDoubleComplex*) fFourier_D, (cufftDoubleReal*)dataPtr_D);
	}
	else if (complexDataPtr_D != NULL)
	{
		cufftExecZ2Z(fPlanBackward_gpu, (cufftDoubleComplex*)fFourier_D, (cufftDoubleComplex*)complexDataPtr_D, CUFFT_INVERSE);
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

void FourierTransformer::setFourier_gpu(cufftDoubleComplex* inputFourier, int nr_images, int xdim, int ydim, int zdim)
{
	cudaMemcpy(fFourier_D, inputFourier,
	           nr_images * zdim * ydim * xdim * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
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

template void windowFourierTransform_gpu<cufftDoubleComplex>(cufftDoubleComplex* in,
                                                             cufftDoubleComplex* out,
                                                             int newdim,
                                                             int nr_images,
                                                             int ndim,
                                                             int xdim,
                                                             int ydim,
                                                             int zdim);
template void windowFourierTransform_gpu<double>(double* in,
                                                 double* out,
                                                 int newdim,
                                                 int nr_images,
                                                 int ndim,
                                                 int xdim,
                                                 int ydim,
                                                 int zdim);

template<typename T>
void selfApplyBeamTilt_gpu(T* Fimg_D, double* beamtilt_x_D, double* beamtilt_y_D,
                           double* wavelength, double* Cs_D, double angpix, int ori_size, int nr_images, int ndim, int xdim, int ydim)
{
	if (ndim != 2)
	{
		REPORT_ERROR("applyBeamTilt can only be done on 2D Fourier Transforms!");
	}
	else
	{
		dim3 dimBlock(BLOCK_X, BLOCK_Y, 1);
		dim3 dimGrid(nr_images, 1, 1);
		double boxsize = angpix * ori_size;
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

template void selfApplyBeamTilt_gpu<cufftDoubleComplex>(cufftDoubleComplex* Fimg_D, double* beamtilt_x_D, double* beamtilt_y_D,
                                                        double* wavelength, double* Cs_D, double angpix, int ori_size, int nr_images, int ndim, int xdim, int ydim);


template<typename T>
static __global__ void shiftImageInFourierTransform_1D_kernel(T* in, T* out,
                                                              double* shift, int nr_images, int nr_trans, int nr_oversampled_trans,  int xdim)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int trans_id = bid % (nr_trans * nr_oversampled_trans);
	double xshift = shift[trans_id];
	T* local_in = in + bid * xdim;
	T* local_out = out + bid * xdim;
	if (abs(xshift) < XMIPP_EQUAL_ACCURACY)
	{
		return;
	}
	int n_loop = (xdim + blockDim.x - 1) / blockDim.x;
	double dotp, a, b, c, d, ac, bd, ab_cd, x;
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
                                                              double* shift, int nr_images, int nr_trans, int nr_oversampled_trans,  int xdim, int ydim, int _nr_elem)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int bid = blockIdx.x;
	int nr_loopx, nr_loopy;
	nr_loopx = (xdim + blockDim.x - 1) / blockDim.x;
	nr_loopy = (ydim + blockDim.y - 1) / blockDim.y;

	int trans_id = bid % (nr_trans * nr_oversampled_trans);
	double xshift = shift[trans_id * 2];
	double yshift = shift[trans_id * 2 + 1];
	T* local_in = in + (bid / (nr_trans * nr_oversampled_trans)) * xdim * ydim;
	T* local_out = out + bid * xdim * ydim;
	if (abs(xshift) < XMIPP_EQUAL_ACCURACY && abs(yshift) < XMIPP_EQUAL_ACCURACY)
	{
		return;
	}
	double dotp, a, b, c, d, ac, bd, ab_cd;
	double sampling = 2 * 3.14159265358979323846 / (double) _nr_elem;
	for (int j = 0; j < nr_loopy; j++)
	{
		double y = (double)tidy + j * blockDim.y;

		if ((tidy + j * blockDim.y) < ydim)
		{
			for (int i = 0; i < nr_loopx; i++)
			{
				double x = (double)tidx + i * blockDim.x;
				if ((tidx + i * blockDim.x) < xdim)
				{
					if (y >= (double)xdim)
					{
						y = y - (double)ydim;
					}
					dotp = 2 * 3.14159265358979323846 * (x * xshift + y * yshift);
					int idx = (int)(ABS(dotp) / sampling) % _nr_elem;
					double xx = ((double)idx * sampling);
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
                                                              double* shift, int nr_images, int nr_trans, int nr_oversampled_trans,  int xdim, int ydim, int zdim)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int bid = blockIdx.x;
	int tid = tidx + tidy * blockDim.x;
	int nr_loopx, nr_loopy;
	nr_loopx = (xdim + blockDim.x - 1) / blockDim.x;
	nr_loopy = (ydim + blockDim.y - 1) / blockDim.y;

	int trans_id = bid % (nr_trans * nr_oversampled_trans);
	double xshift = shift[trans_id * 3];
	double yshift = shift[trans_id * 3 + 1];
	double zshift = shift[trans_id * 3 + 2];
	T* local_in = in + bid * xdim * ydim * zdim;
	T* local_out = out + bid * xdim * ydim * zdim;

	if (abs(xshift) < XMIPP_EQUAL_ACCURACY && abs(yshift) < XMIPP_EQUAL_ACCURACY && abs(zshift) < XMIPP_EQUAL_ACCURACY)
	{
		return;
	}

	double dotp, a, b, c, d, ac, bd, ab_cd, x, y, z;
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
                                      double oridim, double* shift,
                                      int nr_images,  int nr_trans, int nr_oversampled_trans,
                                      int xdim, int ydim, int zdim)
{

	int ndim = (zdim > 1 ? 3 : (ydim > 1 ? 2 : 1));
	int num_shift_imgs = nr_images * nr_trans * nr_oversampled_trans;
	int _nr_elem = 5000;
	double* shift_D;
	double* shift_host;
	shift_host = (double*)malloc(sizeof(double) * nr_trans * nr_oversampled_trans * ndim);
	cudaMalloc((void**)&shift_D, sizeof(double)*nr_trans * nr_oversampled_trans * ndim);
	if (ndim == 1)
	{
		for (int i = 0 ; i < nr_trans * nr_oversampled_trans; i++)
		{
			shift_host[i] = (shift[i] / (-oridim));
		}
		cudaMemcpy(shift_D, shift_host, sizeof(double)*nr_trans * nr_oversampled_trans, cudaMemcpyHostToDevice);
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
		cudaMemcpy(shift_D, shift_host, sizeof(double)*nr_trans * nr_oversampled_trans * ndim, cudaMemcpyHostToDevice);
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
		cudaMemcpy(shift_D, shift_host, sizeof(double)*nr_trans * nr_oversampled_trans * ndim, cudaMemcpyHostToDevice);
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

template void shiftImageInFourierTransform_gpu<cufftDoubleComplex>(cufftDoubleComplex* in,
                                                                   cufftDoubleComplex* out,
                                                                   double oridim, double* shift,
                                                                   int nr_images,  int nr_trans, int nr_oversampled_trans,
                                                                   int xdim, int ydim, int zdim);
