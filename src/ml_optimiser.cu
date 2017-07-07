/***************************************************************************
 *
 * Author : "Huayou SU, Wen WEN, Xiaoli DU, Dongsheng LI"
 * Parallel and Distributed Processing Laboratory of NUDT
 * Author : "Maofu LIAO"
 * Department of Cell Biology, Harvard Medical School
 *
 * This file is GPU host program of GeRelio, which invoke 
 * the kernels to accelerate the expectation. The functions with suffix "_gpu"
 * and simular to their CPU implementaion of file Ml_optimiser.cpp
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


#include "src/ml_optimiser.h"
#include "src/math_function.h"
#include "src/timer.h"

#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <iostream>
#include <stdlib.h>
//#define DEBUG
//#define DEBUG_CHECKSIZES
//#define CHECKSIZES
//Some global threads management variables
extern void centerFFT_2_gpu(DOUBLE* in, DOUBLE* out, int nr_images, int dim, int xdim, int ydim, int zdim, bool forward);

struct not_equal_negtive_999
{
	__host__ __device__ bool operator()(const DOUBLE val) const
	{
		return (bool)(val != (-999.));
	}
};
struct not_equal_zero
{
	__host__ __device__ bool operator()(const DOUBLE val) const
	{
		return (bool)(val != 0.);
	}
};

struct equal_mini_weight
{
	const DOUBLE a;
	equal_mini_weight(DOUBLE _a): a(_a) {}
	__host__ __device__ bool operator()(const DOUBLE val) const
	{
		return (bool)(val == a);
	}
};
struct greater_than
{
	const DOUBLE a;
	greater_than(DOUBLE _a): a(_a) {}
	__host__ __device__ bool operator()(const DOUBLE val) const
	{
		return (bool)(val >= a);
	}
};

struct smaller_than
{
	const DOUBLE a;
	smaller_than(DOUBLE _a): a(_a) {}
	__host__ __device__ bool operator()(const DOUBLE val) const
	{
		return (bool)(val <= a);
	}
};


struct sort_functor
{
	thrust::device_ptr <DOUBLE> data;
	int dsize;
	__host__ __device__
	void operator()(int start_idx)
	{
		thrust::sort(thrust::device, data + (dsize * start_idx), data + (dsize * (start_idx + 1)), thrust::greater<DOUBLE>());
	}
};
void MlOptimiser::project_backproject_GPU_data_init()
{
	//printf("before malloc in cu %p \n",  project_data_D);
	long int project_size = mymodel.nr_classes * (mymodel.PPref[0]).data.zyxdim;
	int yellow_project_size = mymodel.nr_classes * (mymodel.PPref_yellow[0]).data.zyxdim;
	long int total_size = project_size;
	if(sub_extract)
		total_size += project_size +yellow_project_size;
	//cudaMalloc((void**)&project_data_D, mymodel.nr_classes * (mymodel.PPref[0]).data.zyxdim * sizeof(CUFFT_COMPLEX ));
	cudaMalloc((void**)&project_data_D, total_size * sizeof(CUFFT_COMPLEX ));
	//cudaMalloc((void**)&yellow_project_data_D, mymodel.nr_classes * (mymodel.PPref[0]).data.zyxdim * sizeof(CUFFT_COMPLEX ));
	cudaMalloc((void**)&backproject_data_D, wsum_model.nr_classes * (wsum_model.BPref[0]).data.zyxdim * sizeof(CUFFT_COMPLEX ));
	cudaMalloc((void**)&backproject_Weight_D, wsum_model.nr_classes * (wsum_model.BPref[0]).data.zyxdim * sizeof(DOUBLE));
	//if(sub_extract)
	//{
	//	cudaMalloc((void**)&project_data_D+project_size, mymodel.nr_classes * (mymodel.PPref[0]).data.zyxdim * sizeof(CUFFT_COMPLEX ));
	//	cudaMalloc((void**)&yellow_project_data_D+project, mymodel.nr_classes * (mymodel.PPref[0]).data.zyxdim * sizeof(CUFFT_COMPLEX ));
	//}
	
	//printf("after malloc in cu %p\n",  project_data_D);
	for (int i = 0; i < mymodel.nr_classes; i++)
	{
		cudaMemcpy(project_data_D + i * (mymodel.PPref[0]).data.zyxdim, (mymodel.PPref[i]).data.data, (mymodel.PPref[i]).data.zyxdim * 2 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
		if(sub_extract)
		{
			
			cudaMemcpy(project_data_D+project_size + i * (mymodel.PPref_red[0]).data.zyxdim, (mymodel.PPref_red[i]).data.data, (mymodel.PPref_red[i]).data.zyxdim * 2 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
			cudaMemcpy(project_data_D +project_size*2+ i * (mymodel.PPref_yellow[0]).data.zyxdim, (mymodel.PPref_yellow[i]).data.data, (mymodel.PPref_yellow[i]).data.zyxdim * 2 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
		}
	}
	
	cudaMemset(backproject_data_D, 0.,  wsum_model.nr_classes * (wsum_model.BPref[0]).data.zyxdim * sizeof(CUFFT_COMPLEX ));
	cudaMemset(backproject_Weight_D , 0., wsum_model.nr_classes * (wsum_model.BPref[0]).weight.zyxdim * sizeof(DOUBLE));

}

void MlOptimiser::copy_backproject_data_to_CPU()
{
	//copy back the backproject Data to CPU
	for (int i = 0; i < wsum_model.nr_classes; i++)
	{
		cudaMemcpy((wsum_model.BPref[i]).data.data,  backproject_data_D + i * (wsum_model.BPref[0]).data.zyxdim, (wsum_model.BPref[i]).data.zyxdim * sizeof(CUFFT_COMPLEX ), cudaMemcpyDeviceToHost);
		cudaMemcpy((wsum_model.BPref[i]).weight.data, backproject_Weight_D + i * (wsum_model.BPref[0]).weight.zyxdim, (wsum_model.BPref[i]).weight.zyxdim * sizeof(DOUBLE), cudaMemcpyDeviceToHost);

	}
	cudaFree(backproject_data_D);
	cudaFree(backproject_Weight_D);
	cudaFree(project_data_D);
	//if(sub_extract)
	{
		//cudaFree(red_project_data_D);
		//cudaFree(yellow_project_data_D);
	}
}


void MlOptimiser::expectation_gpu()
{

	//#define DEBUG_EXP
#ifdef DEBUG_EXP
	std::cerr << "Entering expectation" << std::endl;
#endif
	// Initialise some stuff
	// A. Update current size (may have been changed to ori_size in autoAdjustAngularSampling) and resolution pointers
	updateImageSizeAndResolutionPointers();
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	device_mem_G = (float) deviceProp.totalGlobalMem / 1048576.0f;
	// B. Initialise Fouriertransform, set weights in wsum_model to zero, etc
	expectationSetup();

	project_backproject_GPU_data_init();
#ifdef DEBUG_EXP
	std::cerr << "Expectation: done setup" << std::endl;
#endif

	// C. Calculate expected minimum angular errors (only for 3D refinements)
	// And possibly update orientational sampling automatically
	// TODO: also implement estimate angular sampling for 3D refinements
	if (!((iter == 1 && do_firstiter_cc) || do_always_cc) && !do_skip_align)
	{
		// Set the exp_metadata (but not the exp_imagedata which is not needed for calculateExpectedAngularErrors)
		int n_trials_acc = (mymodel.ref_dim == 3) ? 100 : 10;
		n_trials_acc = XMIPP_MIN(n_trials_acc, mydata.numberOfOriginalParticles());
		getMetaAndImageDataSubset(0, n_trials_acc - 1, false);
		calculateExpectedAngularErrors(0, n_trials_acc - 1);
	}

	// D. Update the angular sampling (all nodes except master)
	if (iter > 1 && (do_auto_refine))
	{
		updateAngularSampling();
	}

	// E. Check whether everything fits into memory, possibly adjust nr_pool and setup thread task managers
	expectationSetupCheckMemory();

#ifdef DEBUG_EXP
	std::cerr << "Expectation: done setupCheckMemory" << std::endl;
#endif
	if (verb > 0)
	{
		std::cout << " Expectation iteration " << iter;
		if (!do_auto_refine)
		{
			std::cout << " of " << nr_iter;
		}
		std::cout << std::endl;
		init_progress_bar(mydata.numberOfOriginalParticles());
	}

	int barstep = XMIPP_MAX(1, mydata.numberOfOriginalParticles() / 60);
	long int prev_barstep = 0, nr_ori_particles_done = 0;

	// Now perform real expectation over all particles
	// Use local parameters here, as also done in the same overloaded function in MlOptimiserMpi
	long int my_first_ori_particle, my_last_ori_particle;
	//for profiling, we only test for maximal 3 pools

	while (nr_ori_particles_done < mydata.numberOfOriginalParticles())    //should be Parallel with GPU, the workloads of one image of each particle is assigined to one thread block,
	{
		//my_first_ori_particle = 0;//nr_ori_particles_done;
		my_first_ori_particle = nr_ori_particles_done;
		my_last_ori_particle = XMIPP_MIN(mydata.numberOfOriginalParticles() - 1, my_first_ori_particle + nr_pool - 1);  //enlarge the value of nr_pool
		//my_last_ori_particle = mydata.numberOfOriginalParticles()-1;//XMIPP_MIN(mydata.numberOfOriginalParticles() - 1, my_first_ori_particle + mydata.numberOfOriginalParticles() - 1);
		// Get the metadata for these particles
		getMetaAndImageDataSubset(my_first_ori_particle, my_last_ori_particle); //Not parallelized, will be extracted before the loop
		// perform the actual expectation step on several particles
		// GPU function
		expectationSomeParticles_gpu(my_first_ori_particle, my_last_ori_particle);
		setMetaDataSubset(my_first_ori_particle, my_last_ori_particle);
		// Also monitor the changes in the optimal orientations and classes
		monitorHiddenVariableChanges(my_first_ori_particle, my_last_ori_particle);

		nr_ori_particles_done += my_last_ori_particle - my_first_ori_particle + 1;
		if (verb > 0 && nr_ori_particles_done - prev_barstep > barstep)
		{
			prev_barstep = nr_ori_particles_done;
			progress_bar(nr_ori_particles_done);
		}
	}

	copy_backproject_data_to_CPU();

	if (verb > 0)
	{
		progress_bar(mydata.numberOfOriginalParticles());
	}

	// Clean up some memory
	for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
	{
		mymodel.PPref[iclass].data.clear();
	}
#ifdef DEBUG_EXP
	std::cerr << "Expectation: done " << std::endl;
#endif

}

void MlOptimiser::expectationSomeParticles_gpu(long int my_first_ori_particle, long int my_last_ori_particle)
{
#ifdef TIMING
	timer.tic(TIMING_ESP);
#endif

	//#define DEBUG_EXPSINGLE
#ifdef DEBUG_EXPSINGLE
	std::cerr << "Entering expectationSomeParticles..." << std::endl;
#endif

#ifdef TIMING
	timer.tic(TIMING_ESP);
	timer.tic(TIMING_ESP_READ);
#endif

	// Use global variables for thread visibility
	exp_my_first_ori_particle = my_first_ori_particle;
	exp_my_last_ori_particle = my_last_ori_particle;
	exp_nr_ori_particles = exp_my_last_ori_particle - exp_my_first_ori_particle + 1;

	// Find out how many particles there are in these ori_particles
	exp_nr_particles = 0;
	for (long int i = my_first_ori_particle; i <= my_last_ori_particle; i++)
	{
		exp_nr_particles += mydata.ori_particles[i].particles_id.size();
	}

	// If there are more than one particle in each ori_particle, then do these in parallel with threads
	if (nr_pool == 1 && exp_nr_particles / exp_nr_ori_particles > 1)
	{
		int my_pool = exp_nr_particles / exp_nr_ori_particles;
		exp_ipart_ThreadTaskDistributor->resize(my_pool, 1);
	}

	// TODO: MAKE SURE THAT ALL PARTICLES IN SomeParticles ARE FROM THE SAME AREA, SO THAT THE R_mic CAN BE RE_USED!!!

	// In the first iteration, multiple seeds will be generated
	// A single random class is selected for each pool of images, and one does not marginalise over the orientations
	// The optimal orientation is based on signal-product (rather than the signal-intensity sensitive Gaussian)
	// If do_firstiter_cc, then first perform a single iteration with K=1 and cross-correlation criteria, afterwards

	// Generally: use all references
	iclass_min = 0;
	iclass_max = mymodel.nr_classes - 1;
	// low-pass filter again and generate the seeds
	if (do_generate_seeds)
	{
		if (do_firstiter_cc && iter == 1)
		{
			// In first (CC) iter, use a single reference (and CC)
			iclass_min = iclass_max = 0;
		}
		else if ((do_firstiter_cc && iter == 2) || (!do_firstiter_cc && iter == 1))
		{
			// In second CC iter, or first iter without CC: generate the seeds
			// Now select a single random class
			// exp_part_id is already in randomized order (controlled by -seed)
			// WARNING: USING SAME iclass_min AND iclass_max FOR SomeParticles!!
			iclass_min = iclass_max = divide_equally_which_group(mydata.numberOfOriginalParticles(), mymodel.nr_classes, exp_my_first_ori_particle);
		}
	}

	// TODO: think of a way to have the different images in a single series have DIFFERENT offsets!!!
	// Right now, they are only centered with a fixed relative translation!!!!

	// Thid debug is a good one to step through the separate steps of the expectation to see where trouble lies....
	//#define DEBUG_ESP_MEM
#ifdef DEBUG_ESP_MEM
	char c;
	std::cerr << "Before getFourierTransformsAndCtfs, press any key to continue... " << std::endl;
	std::cin >> c;
#endif

	// Read all image of this series into memory, apply old origin offsets and store Fimg, Fctf, exp_old_xoff and exp_old_yoff in vectors./
	//The GPU function for GetFourierTransformsAndCtfs, do not use the pthread programming
	doThreadGetFourierTransformsAndCtfs_gpu();
	if (do_realign_movies) //&& movie_frame_running_avg_side > 0)
	{
		calculateRunningAveragesOfMovieFrames();
	}

#ifdef DEBUG_ESP_MEM
	std::cerr << "After getFourierTransformsAndCtfs, press any key to continue... " << std::endl;
	std::cin >> c;
#endif

#ifdef TIMING
	timer.toc(TIMING_ESP_READ);
#endif

	// Initialise significant weight to minus one, so that all coarse sampling points will be handled in the first pass
	exp_significant_weight.clear();
	exp_significant_weight.resize(exp_nr_particles);
	for (int n = 0; n < exp_nr_particles; n++)
	{
		exp_significant_weight[n] = -1.;
	}

	// Number of rotational and translational sampling points
	exp_nr_trans = sampling.NrTranslationalSamplings();

	exp_nr_dir = sampling.NrDirections();
	exp_nr_psi = sampling.NrPsiSamplings();
	exp_nr_rot = exp_nr_dir * exp_nr_psi;
	// Only perform a second pass when using adaptive oversampling
	int nr_sampling_passes = (adaptive_oversampling > 0) ? 2 : 1;
	// Pass twice through the sampling of the entire space of rot, tilt and psi
	// The first pass uses a coarser angular sampling and possibly smaller FFTs than the second pass.
	// Only those sampling points that contribute to the highest x% of the weights in the first pass are oversampled in the second pass
	// Only those sampling points will contribute to the weighted sums in the third loop below
	
	for (exp_ipass = 0; exp_ipass < nr_sampling_passes; exp_ipass++)
	{

		if (strict_highres_exp > 0.)
			// Use smaller images in both passes and keep a maximum on coarse_size, just like in FREALIGN
		{
			exp_current_image_size = coarse_size;
		}
		else if (adaptive_oversampling > 0)
			// Use smaller images in the first pass, larger ones in the second pass
		{
			exp_current_image_size = (exp_ipass == 0) ? coarse_size : mymodel.current_size;
		}
		else
		{
			exp_current_image_size = mymodel.current_size;
		}

		// Use coarse sampling in the first pass, oversampled one the second pass
		exp_current_oversampling = (exp_ipass == 0) ? 0 : adaptive_oversampling;
		exp_nr_oversampled_rot = sampling.oversamplingFactorOrientations(exp_current_oversampling);
		exp_nr_oversampled_trans = sampling.oversamplingFactorTranslations(exp_current_oversampling);


#ifdef DEBUG_ESP_MEM

		std::cerr << "Before getAllSquaredDifferences, use top to see memory usage and then press any key to continue... " << std::endl;
		std::cin >> c;
#endif
		// Calculate the squared difference terms inside the Gaussian kernel for all hidden variables
		// GPU function
		
		getAllSquaredDifferences_gpu();
#ifdef DEBUG_ESP_MEM
		std::cerr << "After getAllSquaredDifferences, use top to see memory usage and then press any key to continue... " << std::endl;
		std::cin >> c;
#endif

		// Now convert the squared difference terms to weights,
		// also calculate exp_sum_weight, and in case of adaptive oversampling also exp_significant_weight

		// GPU function
		convertAllSquaredDifferencesToWeights_gpu();
		//Release the GPU memory for the first pass
		if (exp_ipass == 0)
		{
			cudaFree(exp_local_sqrtXi2_D);
			cudaFree(exp_local_Fimgs_shifted_D);
			cudaFree(exp_Minvsigma2s_D);
			cudaFree(exp_local_Fctfs_D);
			cudaFree(exp_Mweight_D);
		}

		cudaFree(exp_min_diff2_D);

#ifdef DEBUG_ESP_MEM
		std::cerr << "After convertAllSquaredDifferencesToWeights, press any key to continue... " << std::endl;
		std::cin >> c;
#endif

	}// end loop over 2 exp_ipass iterations

	exp_current_image_size = mymodel.current_size;

#ifdef DEBUG_ESP_MEM
	std::cerr << "Before storeWeightedSums, press any key to continue... " << std::endl;
	std::cin >> c;
#endif
	// GPU function
	storeWeightedSums_gpu();
	cudaFree(exp_local_sqrtXi2_D);
	cudaFree(exp_local_Fimgs_shifted_D);
	cudaFree(exp_local_Fimgs_shifted_nomask_D);
	cudaFree(exp_Minvsigma2s_D);
	cudaFree(exp_local_Fctfs_D);
	cudaFree(exp_Mweight_D);


#ifdef DEBUG_ESP_MEM
	std::cerr << "After storeWeightedSums, press any key to continue... " << std::endl;
	std::cin >> c;
#endif
#ifdef DEBUG_EXPSINGLE
	std::cerr << "Leaving expectationSingleParticle..." << std::endl;
#endif

#ifdef TIMING
	timer.toc(TIMING_ESP);
#endif

}
//void MlOptimiser:: subYellowMK_gpu( DOUBLE *image_red_D)

void MlOptimiser::doThreadGetFourierTransformsAndCtfs_gpu()
/*{

	exp_starting_image_no.clear();
	exp_power_imgs.clear();
	exp_highres_Xi2_imgs.clear();
	exp_Fimgs.clear();
	exp_Fimgs_nomask.clear();
	exp_Fctfs.clear();
	exp_old_offset.clear();
	exp_prior.clear();
	exp_local_oldcc.clear();
	exp_ipart_to_part_id.clear();
	exp_ipart_to_ori_part_id.clear();
	exp_ipart_to_ori_part_nframe.clear();
	exp_iimg_to_ipart.clear();

	// Resize to the right size instead of using pushbacks
	exp_starting_image_no.resize(exp_nr_particles);
	// First check how many images there are in the series for each particle...
	// And calculate exp_nr_images
	exp_nr_images = 0;
	for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{

#ifdef DEBUG_CHECKSIZES
		if (ori_part_id >= mydata.ori_particles.size())
		{
			std::cerr << "ori_part_id= " << ori_part_id << " mydata.ori_particles.size()= " << mydata.ori_particles.size() << std::endl;
			REPORT_ERROR("ori_part_id >= mydata.ori_particles.size()");
		}
#endif
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			int iipart = exp_ipart_to_part_id.size();
			exp_starting_image_no.at(iipart) = exp_nr_images;
			exp_nr_images += mydata.getNrImagesInSeries(part_id);
			exp_ipart_to_part_id.push_back(part_id);
			exp_ipart_to_ori_part_id.push_back(ori_part_id);
			exp_ipart_to_ori_part_nframe.push_back(i);
			for (int i = 0; i < mydata.getNrImagesInSeries(part_id); i++)
			{
				exp_iimg_to_ipart.push_back(iipart);
			}
		}
	}
	// Then also resize vectors for all images
	exp_power_imgs.resize(exp_nr_images);
	exp_highres_Xi2_imgs.resize(exp_nr_images);
	exp_Fimgs.resize(exp_nr_images);
	exp_Fimgs_nomask.resize(exp_nr_images);
	exp_Fctfs.resize(exp_nr_images);
	exp_old_offset.resize(exp_nr_images);
	exp_prior.resize(exp_nr_images);
	exp_local_oldcc.resize(exp_nr_images);

	FourierTransformer transformer;

	DOUBLE* normcorr_H;
	DOUBLE* beamtilt_x_H, *beamtilt_y_H;
	DOUBLE* Cs_H, *lambda_H;
	//DOUBLE *translated_local_Matrixs;
	DOUBLE* exp_old_offset_H, *exp_old_offset_D;

	normcorr_H = new DOUBLE[exp_nr_images];
	beamtilt_x_H = new DOUBLE[exp_nr_images];
	beamtilt_y_H = new DOUBLE[exp_nr_images];
	Cs_H = new DOUBLE[exp_nr_images];
	lambda_H = new DOUBLE[exp_nr_images];
	exp_old_offset_H = new DOUBLE[exp_nr_images * 2];
	//translated_local_Matrixs = new DOUBLE[exp_nr_images*3*3]; //The image is 2D
	for (long int ipart = 0; ipart < exp_nr_images; ipart++)
	{
		int part_id = exp_ipart_to_part_id[ipart];
		for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
		{
			int my_image_no = exp_starting_image_no.at(ipart) + iseries;
			// Which group do I belong?
			int group_id = mydata.getGroupId(part_id, iseries);

			// Get the norm_correction
			normcorr_H[my_image_no] = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM);

			// Get the optimal origin offsets from the previous iteration
			Matrix1D<DOUBLE> my_old_offset(2), my_prior(2);
			XX(my_old_offset) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF);
			YY(my_old_offset) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF);
			XX(my_prior)      = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF_PRIOR);
			YY(my_prior)      = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF_PRIOR);

			// Uninitialised priors were set to 999.
			if (XX(my_prior)   > 998.99 && XX(my_prior)   < 999.01)
			{
				XX(my_prior)   = 0.;
			}
			if (YY(my_prior) > 998.99 && YY(my_prior) < 999.01)
			{
				YY(my_prior) = 0.;
			}

			// Get the old cross-correlations
			exp_local_oldcc.at(my_image_no) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_DLL);

			// If we do local angular searches, get the previously assigned angles to center the prior
			// Only do this for the first image in the series, as this prior work per-particle, not per-image
			// All images in the series use the same rotational sampling, brought back to "exp_R_mic=identity"
			if (do_skip_align || do_skip_rotate)
			{
				// No need to block the threads global_mutex, as nr_pool will be set to 1 anyway for do_skip_align!
				if (do_skip_align)
				{
					// Rounded translations will be applied to the image upon reading,
					// set the unique translation in the sampling object to the fractional difference
					Matrix1D<DOUBLE> rounded_offset = my_old_offset;
					rounded_offset.selfROUND();
					rounded_offset = my_old_offset - rounded_offset;
					sampling.setOneTranslation(rounded_offset);
				}

				// Also set the rotations
				DOUBLE old_rot, old_tilt, old_psi;
				old_rot = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT);
				old_tilt = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT);
				old_psi = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI);
				sampling.setOneOrientation(old_rot, old_tilt, old_psi);

			}
			else if (mymodel.orientational_prior_mode != NOPRIOR && iseries == 0)
			{
				// First try if there are some fixed prior angles
				DOUBLE prior_rot = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT_PRIOR);
				DOUBLE prior_tilt = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT_PRIOR);
				DOUBLE prior_psi = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI_PRIOR);

				// If there were no defined priors (i.e. their values were 999.), then use the "normal" angles
				if (prior_rot > 998.99 && prior_rot < 999.01)
				{
					prior_rot = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT);
				}
				if (prior_tilt > 998.99 && prior_tilt < 999.01)
				{
					prior_tilt = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT);
				}
				if (prior_psi > 998.99 && prior_psi < 999.01)
				{
					prior_psi = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI);
				}

				// For tilted series: convert the angles back onto the untilted ones...
				// Calculate the angles back from the Euler matrix because for tilt series exp_R_mic may have changed them...
				Matrix2D<DOUBLE> A, R_mic(3, 3);
				R_mic(0, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_0);
				R_mic(0, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_1);
				R_mic(0, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_2);
				R_mic(1, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_0);
				R_mic(1, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_1);
				R_mic(1, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_2);
				R_mic(2, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_0);
				R_mic(2, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_1);
				R_mic(2, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_2);
				if (!R_mic.isIdentity())
				{
					Euler_angles2matrix(prior_rot, prior_tilt, prior_psi, A);
					A = R_mic.inv() * A;
					Euler_matrix2angles(A, prior_rot, prior_tilt, prior_psi);
				}

				// Select only those orientations that have non-zero prior probability
				sampling.selectOrientationsWithNonZeroPriorProbability(prior_rot, prior_tilt, prior_psi,
				                                                       sqrt(mymodel.sigma2_rot), sqrt(mymodel.sigma2_tilt), sqrt(mymodel.sigma2_psi));

				long int nr_orients = sampling.NrDirections() * sampling.NrPsiSamplings();
				if (nr_orients == 0)
				{
					std::cerr << " sampling.NrDirections()= " << sampling.NrDirections() << " sampling.NrPsiSamplings()= " << sampling.NrPsiSamplings() << std::endl;
					REPORT_ERROR("Zero orientations fall within the local angular search. Increase the sigma-value(s) on the orientations!");
				}
			}

			my_old_offset.selfROUND();

			exp_old_offset.at(my_image_no) = my_old_offset;
			// Also store priors on translations
			exp_prior.at(my_image_no) = my_prior;
			//Only for copy the offset of each image to GPU by using a large array
			exp_old_offset_H[my_image_no * 2] = XX(my_old_offset);
			exp_old_offset_H[my_image_no * 2 + 1] = YY(my_old_offset);

			beamtilt_x_H[my_image_no] = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_BEAMTILT_X);
			beamtilt_y_H[my_image_no] = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_BEAMTILT_Y);
			Cs_H[my_image_no] = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_CS);
			DOUBLE V = 1000. * DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_VOLTAGE);
			lambda_H[my_image_no] = 12.2643247 / sqrt(V * (1. + V * 0.978466e-6));

		}
	}
	std::vector< Image<DOUBLE> > local_images, local_rec_images;
	std::vector< MultidimArray<Complex> > local_Fimgs, local_Faux;
	std::vector< MultidimArray<DOUBLE> > local_Fctf_images, local_img_aux;

	local_images.clear();
	local_rec_images.clear();
	local_Fimgs.clear();
	local_Faux.clear();
	local_Fctf_images.clear();
	local_img_aux.clear();

	local_images.resize(exp_nr_images);
	local_rec_images.resize(exp_nr_images);
	local_Fimgs.resize(exp_nr_images);
	local_Faux.resize(exp_nr_images);
	local_Fctf_images.resize(exp_nr_images);
	local_img_aux.resize(exp_nr_images);

	DOUBLE* translated_matrix_H;
	int matrix_size = 3 * 3;
	translated_matrix_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles * matrix_size);
	int image_size = mymodel.ori_size * mymodel.ori_size;
	cudaMalloc((void**)&image_D, exp_nr_images * image_size * sizeof(DOUBLE));
	cudaMalloc((void**)&rec_image_D, exp_nr_images * image_size * sizeof(DOUBLE));
	cudaMemcpy(image_D, exp_imagedata.data, sizeof(DOUBLE) *mymodel.ori_size * mymodel.ori_size * exp_nr_images,  cudaMemcpyHostToDevice);
	DOUBLE* normcorr_D;
	cudaMalloc((void**)&normcorr_D, exp_nr_images * sizeof(DOUBLE));
	cudaMemcpy(normcorr_D, normcorr_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	if (do_norm_correction)
	{
		do_norm_correction_gpu(image_D, normcorr_D, image_size, exp_nr_images,  mymodel.avg_norm_correction);
	}

	cudaError cu_error1 = cudaGetLastError();
	if (cu_error1 != cudaSuccess)
	{
		std::cout << "cudaMemcpy Error :" << __LINE__ << std::endl;
		REPORT_ERROR("cudaMemcpy lambda_D ");
	}
	for (long int ipart = 0; ipart < exp_nr_particles; ipart++)
	{
		long int part_id = exp_ipart_to_part_id[ipart];

		// Prevent movies and series at the same time...
		if (mydata.getNrImagesInSeries(part_id) > 1 && do_realign_movies)
		{
			REPORT_ERROR("Not ready yet for dealing with image series at the same time as realigning movie frames....");
		}

		for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
		{
			int my_image_no = exp_starting_image_no.at(ipart) + iseries;
			// Which group do I belong?
			//int group_id = mydata.getGroupId(part_id, iseries);
			// Apply the norm_correction term


			Matrix2D< DOUBLE > tmp;
			translation2DMatrix(exp_old_offset.at(my_image_no), tmp);

			tmp = tmp.inv();
			memcpy(translated_matrix_H + ipart * matrix_size, tmp.mdata, sizeof(DOUBLE)* matrix_size);

		}
	}
	cudaMalloc((void**)&exp_old_offset_D,  exp_nr_images * 2 * sizeof(DOUBLE));
	cudaMalloc((void**)&translated_matrix_D,  exp_nr_images * matrix_size * sizeof(DOUBLE));
	cudaMemcpy(translated_matrix_D, translated_matrix_H, exp_nr_images * matrix_size * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cudaMemcpy(exp_old_offset_D, exp_old_offset_H, exp_nr_images * 2 * sizeof(DOUBLE), cudaMemcpyHostToDevice);

	selfTranslate_gpu(image_D, translated_matrix_D, exp_old_offset_D, 2, mymodel.ori_size, mymodel.ori_size, image_size, exp_nr_images, DONT_WRAP);

	if (has_converged && do_use_reconstruct_images)
	{
		cudaMemcpy(rec_image_D, exp_imagedata.data + exp_nr_images * YXSIZE(exp_imagedata), sizeof(DOUBLE) *mymodel.ori_size * mymodel.ori_size * exp_nr_images, cudaMemcpyHostToDevice);
		selfTranslate_gpu(rec_image_D, translated_matrix_D, exp_old_offset_D, 2, mymodel.ori_size, mymodel.ori_size, image_size, exp_nr_images, DONT_WRAP);
	}

	free(translated_matrix_H);
	cudaFree(normcorr_D);
	cudaFree(exp_old_offset_D);
	//The GPU codes correspond to the above CXX codes, copy images from exp_imagedata
	long int size_ = 0 ;
	image_size = mymodel.ori_size * mymodel.ori_size;
	size_ = exp_nr_images * image_size;
	//TODO: Apply the norm_correction term

	DOUBLE* beamtilt_x_D, *beamtilt_y_D;
	DOUBLE* Cs_D, *lambda_D;


	cudaMalloc((void**)&beamtilt_x_D, exp_nr_images * sizeof(DOUBLE));
	cudaMalloc((void**)&beamtilt_y_D, exp_nr_images * sizeof(DOUBLE));
	cudaMalloc((void**)&Cs_D, exp_nr_images * sizeof(DOUBLE));
	cudaMalloc((void**)&lambda_D, exp_nr_images * sizeof(DOUBLE));

	cudaError cu_error;

	cu_error = cudaMemcpy(beamtilt_x_D, beamtilt_x_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cu_error = cudaMemcpy(beamtilt_y_D, beamtilt_y_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cu_error = cudaMemcpy(Cs_D, Cs_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cu_error = cudaMemcpy(lambda_D, lambda_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	if (cu_error != cudaSuccess)
	{
		std::cout << "cudaMemcpy Error :" << __LINE__ << std::endl;
		REPORT_ERROR("cudaMemcpy lambda_D ");
	}
	DOUBLE* raw_data_images, *raw_rec_images;
	raw_data_images = new DOUBLE[size_];
	if (has_converged && do_use_reconstruct_images)
	{
		raw_rec_images = new DOUBLE[size_];
	}

	if (cu_error != cudaSuccess)
	{
		std::cout << "cudaMemcpy Error :" << __LINE__ << std::endl;
		REPORT_ERROR("cudaMemcpy lambda_D ");
	}

	int dim, xdim, ydim, zdim;

	xdim = mymodel.ori_size;
	ydim = mymodel.ori_size;
	zdim  = 1;
	dim = 2 ;
	//Do the CenterFFT for some partiles with GPU
	DOUBLE* local_images_D;
	cudaMalloc((void**)&local_images_D,  size_ * sizeof(DOUBLE));
	cudaMemset(local_images_D, 0, size_ * sizeof(DOUBLE));
	centerFFT_gpu(((has_converged && do_use_reconstruct_images) ? rec_image_D : image_D) , local_images_D, exp_nr_images, dim, xdim,  ydim, zdim, true);

	cudaError cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}
	//Foutier Transform for some particles on GPU
	CUFFT_COMPLEX * local_Faux_D;
	cudaDeviceSynchronize();

	int ndim = 3;
	if (zdim == 1)
	{
		ndim = 2;
		if (ydim == 1)
		{
			ndim = 1;
		}
	}
	int* N;
	N = (int*)malloc(sizeof(int) * ndim);
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
	cufftHandle fPlanForward_gpu;
#ifdef FLOAT_PRECISION
	cufftResult fftplan1 = cufftPlanMany(&fPlanForward_gpu, ndim, N,
	                                     NULL, 1, 0,
	                                     NULL, 1, 0,
	                                     CUFFT_R2C, exp_nr_images);
	cudaMalloc((void**)&local_Faux_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1)*sizeof(CUFFT_COMPLEX ));
	cufftExecR2C(fPlanForward_gpu, local_images_D, local_Faux_D);
#else
	cufftResult fftplan1 = cufftPlanMany(&fPlanForward_gpu, ndim, N,
	                                     NULL, 1, 0,
	                                     NULL, 1, 0,
	                                     CUFFT_D2Z, exp_nr_images);
	cudaMalloc((void**)&local_Faux_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1)*sizeof(CUFFT_COMPLEX ));
	cufftExecD2Z(fPlanForward_gpu, local_images_D, local_Faux_D);
#endif
	
	ScaleComplexPointwise_gpu(local_Faux_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1), 1.0 / (zdim * ydim * xdim));

	int newhdim = mymodel.current_size / 2 + 1;
	int newdim = mymodel.current_size;
	int mem_size;
	if (ndim == 1)
	{
		mem_size = newhdim * 2 * sizeof(DOUBLE);
	}
	else if (ndim == 2)
	{
		mem_size = newdim * newhdim * 2 * sizeof(DOUBLE);
	}
	else if (ndim == 3)
	{
		mem_size = newdim * newdim * newhdim * 2 * sizeof(DOUBLE);
	}
	cudaMalloc((void**)&exp_Fimgs_nomask_D, exp_nr_images * mem_size);
	windowFourierTransform_gpu(local_Faux_D,
	                           exp_Fimgs_nomask_D,
	                           mymodel.current_size,
	                           exp_nr_images,
	                           2,
	                           (xdim / 2) + 1,
	                           ydim,
	                           zdim);

	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}

	selfApplyBeamTilt_gpu(exp_Fimgs_nomask_D, beamtilt_x_D, beamtilt_y_D,
	                      lambda_D, Cs_D, mymodel.pixel_size, mymodel.ori_size, exp_nr_images,  2,  newhdim,  newdim);
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}
	DOUBLE* Mnoise_D;
	Mnoise_D = NULL;
	softMaskOutsideMap_gpu(image_D, particle_diameter / (2. * mymodel.pixel_size), (DOUBLE)width_mask_edge, Mnoise_D, exp_nr_images, xdim, ydim, zdim);
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}

	centerFFT_gpu(image_D , local_images_D, exp_nr_images, dim, xdim,  ydim, zdim, true);
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}
#ifdef FLOAT_PRECISION
	cufftExecR2C(fPlanForward_gpu, local_images_D, local_Faux_D);
#else
	cufftExecD2Z(fPlanForward_gpu, local_images_D, local_Faux_D);
#endif
	ScaleComplexPointwise_gpu(local_Faux_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1), 1.0 / (zdim * ydim * xdim));
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();
	cufftDestroy(fPlanForward_gpu);
	DOUBLE* exp_highres_Xi2_imgs_H, *exp_highres_Xi2_imgs_D;
	DOUBLE* exp_power_imgs_H, *exp_power_imgs_D;
	exp_highres_Xi2_imgs_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles);
	exp_power_imgs_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles * (mymodel.ori_size / 2 + 1));
	cudaMalloc((void**)&exp_highres_Xi2_imgs_D, exp_nr_particles * sizeof(DOUBLE));
	cudaMemset(exp_highres_Xi2_imgs_D, 0.0, exp_nr_particles * sizeof(DOUBLE));
	cudaMalloc((void**)&exp_power_imgs_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMemset(exp_power_imgs_D, 0.0, exp_nr_particles * (mymodel.ori_size / 2 + 1) * sizeof(DOUBLE));
	calculate_img_power_gpu(local_Faux_D,
	                        exp_power_imgs_D,
	                        exp_highres_Xi2_imgs_D,
	                        exp_nr_images,
	                        (xdim / 2 + 1),
	                        ydim,
	                        zdim,
	                        mymodel.ori_size,
	                        mymodel.current_size
	                       );

	if (mymodel.current_size < mymodel.ori_size)
	{
		cudaMemcpy(exp_power_imgs_H, exp_power_imgs_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);
		cudaMemcpy(exp_highres_Xi2_imgs_H, exp_highres_Xi2_imgs_D, exp_nr_particles * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
		for (long int ipart = 0; ipart < exp_nr_particles; ipart++)
		{
			exp_highres_Xi2_imgs.at(ipart) = exp_highres_Xi2_imgs_H[ipart];
			exp_power_imgs[ipart].resize((mymodel.ori_size / 2 + 1));
			memcpy(exp_power_imgs[ipart].data, exp_power_imgs_H + ipart * (mymodel.ori_size / 2 + 1), (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
		}
	}
	else
	{
		for (long int ipart = 0; ipart < exp_nr_particles; ipart++)
		{
			exp_highres_Xi2_imgs.at(ipart) = 0.;
		}
	}
	cudaFree(exp_power_imgs_D);
	cudaFree(exp_highres_Xi2_imgs_D);
	free(exp_power_imgs_H);
	free(exp_highres_Xi2_imgs_H);
	cudaMalloc((void**)&exp_Fimgs_D, exp_nr_images * mem_size);
	windowFourierTransform_gpu(local_Faux_D,
	                           exp_Fimgs_D,
	                           mymodel.current_size,
	                           exp_nr_images,
	                           2,
	                           (xdim / 2) + 1,
	                           ydim,
	                           zdim);
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}

	//Fourier_image_size = newdim * newhdim;

	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
		exit(EXIT_FAILURE);
	}

	DOUBLE* ctf_related_parameters_H;
	int nr_parameters = 9;
	ctf_related_parameters_H = (DOUBLE*) malloc(nr_parameters * exp_nr_images * sizeof(DOUBLE));
	for (int im = 0; im < exp_nr_images; im++)
	{
		DOUBLE DeltafU = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_DEFOCUS_U);
		DOUBLE DeltafV  = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_DEFOCUS_V);
		DOUBLE azimuthal_angle  = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_DEFOCUS_ANGLE);
		DOUBLE kV   = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_VOLTAGE);
		DOUBLE Cs   = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_CS);
		DOUBLE Q0   = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_Q0);
		DOUBLE Bfac = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_BFAC);
		DOUBLE scale = 1.0;
		DOUBLE local_Cs = Cs * 1e7;
		DOUBLE local_kV = kV * 1e3;
		DOUBLE rad_azimuth = DEG2RAD(azimuthal_angle);

		DOUBLE defocus_average   = -(DeltafU + DeltafV) * 0.5;
		DOUBLE defocus_deviation = -(DeltafU - DeltafV) * 0.5;

		DOUBLE lambda = 12.2643247 / sqrt(local_kV * (1. + local_kV * 0.978466e-6));

		DOUBLE K1 = PI / 2 * 2 * lambda;
		DOUBLE K2 = PI / 2 * local_Cs * lambda * lambda * lambda;
		DOUBLE K3 = sqrt(1 - Q0 * Q0);
		DOUBLE K4 = -Bfac / 4.;
		ctf_related_parameters_H[im * nr_parameters + 0] = defocus_average;
		ctf_related_parameters_H[im * nr_parameters + 1] = defocus_deviation;
		ctf_related_parameters_H[im * nr_parameters + 2] = rad_azimuth;
		ctf_related_parameters_H[im * nr_parameters + 3] = Q0;
		ctf_related_parameters_H[im * nr_parameters + 4] = scale;
		ctf_related_parameters_H[im * nr_parameters + 5] = K1;
		ctf_related_parameters_H[im * nr_parameters + 6] = K2;
		ctf_related_parameters_H[im * nr_parameters + 7] = K3;
		ctf_related_parameters_H[im * nr_parameters + 8] = K4;

	}
	cudaMalloc((void**)&exp_Fctf_D, exp_nr_images * newdim * newhdim * sizeof(DOUBLE));
	getFftwImage_gpu(exp_Fctf_D, ctf_related_parameters_H,
	                 mymodel.ori_size, mymodel.ori_size, mymodel.pixel_size,
	                 ctf_phase_flipped, only_flip_phases, intact_ctf_first_peak, true,
	                 exp_nr_images,
	                 newhdim,
	                 newdim,
	                 do_ctf_correction);

	free(ctf_related_parameters_H);
	cudaDeviceSynchronize();
	for (long int ipart = 0; ipart < exp_nr_particles; ipart++)
	{
		long int part_id = exp_ipart_to_part_id[ipart];

		for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
		{
			int my_image_no = exp_starting_image_no.at(ipart) + iseries;


			// Store Fimg and Fctf

			exp_Fimgs[my_image_no].resize(1, newdim, newhdim);
			exp_Fimgs_nomask[my_image_no].resize(1, newdim, newhdim);

			exp_Fctfs[my_image_no].resize(1, newdim, newhdim);
		} // end loop iseries
	}// end loop ipart
	transformer.free_memory_gpu();
	transformer.cleanup();

	cudaFree(translated_matrix_D);
	cudaFree(beamtilt_x_D);
	cudaFree(beamtilt_y_D);
	cudaFree(Cs_D);
	cudaFree(lambda_D);
	cudaFree(image_D);
	cudaFree(rec_image_D);
	cudaFree(local_images_D);
	cudaFree(local_Faux_D);

	delete [] normcorr_H;
	delete [] beamtilt_x_H;
	delete [] beamtilt_y_H;
	delete [] Cs_H;
	delete [] lambda_H;
	delete []exp_old_offset_H;
	delete [] raw_data_images;
	free(N);
	if (has_converged && do_use_reconstruct_images)
	{
		delete []raw_rec_images;
	}


}
*/
		{
		
			exp_starting_image_no.clear();
			exp_power_imgs.clear();
			exp_highres_Xi2_imgs.clear();
			exp_Fimgs.clear();
			exp_Fimgs_nomask.clear();
			exp_Fctfs.clear();
			exp_old_offset.clear();
			exp_prior.clear();
			exp_local_oldcc.clear();
			exp_ipart_to_part_id.clear();
			exp_ipart_to_ori_part_id.clear();
			exp_ipart_to_ori_part_nframe.clear();
			exp_iimg_to_ipart.clear();
		
			
			exp_highres_Xi2_imgs_red.clear();
			exp_power_imgs_red.clear();
			// Resize to the right size instead of using pushbacks
			exp_starting_image_no.resize(exp_nr_particles);
			// First check how many images there are in the series for each particle...
			// And calculate exp_nr_images
			exp_nr_images = 0;
			for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
			{
		
#ifdef DEBUG_CHECKSIZES
				if (ori_part_id >= mydata.ori_particles.size())
				{
					std::cerr << "ori_part_id= " << ori_part_id << " mydata.ori_particles.size()= " << mydata.ori_particles.size() << std::endl;
					REPORT_ERROR("ori_part_id >= mydata.ori_particles.size()");
				}
#endif
				for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
				{
					long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
					int iipart = exp_ipart_to_part_id.size();
					exp_starting_image_no.at(iipart) = exp_nr_images;
					exp_nr_images += mydata.getNrImagesInSeries(part_id);
					exp_ipart_to_part_id.push_back(part_id);
					exp_ipart_to_ori_part_id.push_back(ori_part_id);
					exp_ipart_to_ori_part_nframe.push_back(i);
					for (int i = 0; i < mydata.getNrImagesInSeries(part_id); i++)
					{
						exp_iimg_to_ipart.push_back(iipart);
					}
				}
			}
			// Then also resize vectors for all images
			exp_power_imgs.resize(exp_nr_images);
			exp_highres_Xi2_imgs.resize(exp_nr_images);
			exp_Fimgs.resize(exp_nr_images);
			exp_Fimgs_nomask.resize(exp_nr_images);
			exp_Fctfs.resize(exp_nr_images);
			exp_old_offset.resize(exp_nr_images);
			exp_prior.resize(exp_nr_images);
			exp_local_oldcc.resize(exp_nr_images);
		
			exp_power_imgs_red.resize(exp_nr_images);
			exp_highres_Xi2_imgs_red.resize(exp_nr_images);
		
			FourierTransformer transformer;
		
			DOUBLE* normcorr_H;
			DOUBLE* beamtilt_x_H, *beamtilt_y_H;
			DOUBLE* Cs_H, *lambda_H;
			//DOUBLE *translated_local_Matrixs;
			DOUBLE* exp_old_offset_H, *exp_old_offset_D;
		
			normcorr_H = new DOUBLE[exp_nr_images];
			beamtilt_x_H = new DOUBLE[exp_nr_images];
			beamtilt_y_H = new DOUBLE[exp_nr_images];
			Cs_H = new DOUBLE[exp_nr_images];
			lambda_H = new DOUBLE[exp_nr_images];
			exp_old_offset_H = new DOUBLE[exp_nr_images * 2];
			//translated_local_Matrixs = new DOUBLE[exp_nr_images*3*3]; //The image is 2D
			for (long int ipart = 0; ipart < exp_nr_images; ipart++)
			{
				int part_id = exp_ipart_to_part_id[ipart];
				for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
				{
					int my_image_no = exp_starting_image_no.at(ipart) + iseries;
					// Which group do I belong?
					int group_id = mydata.getGroupId(part_id, iseries);
		
					// Get the norm_correction
					normcorr_H[my_image_no] = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM);
		
					// Get the optimal origin offsets from the previous iteration
					Matrix1D<DOUBLE> my_old_offset(2), my_prior(2);
					XX(my_old_offset) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF);
					YY(my_old_offset) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF);
					XX(my_prior)	  = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF_PRIOR);
					YY(my_prior)	  = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF_PRIOR);
		
					// Uninitialised priors were set to 999.
					if (XX(my_prior)   > 998.99 && XX(my_prior)   < 999.01)
					{
						XX(my_prior)   = 0.;
					}
					if (YY(my_prior) > 998.99 && YY(my_prior) < 999.01)
					{
						YY(my_prior) = 0.;
					}
		
					// Get the old cross-correlations
					exp_local_oldcc.at(my_image_no) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_DLL);
		
					// If we do local angular searches, get the previously assigned angles to center the prior
					// Only do this for the first image in the series, as this prior work per-particle, not per-image
					// All images in the series use the same rotational sampling, brought back to "exp_R_mic=identity"
					if (do_skip_align || do_skip_rotate)
					{
						// No need to block the threads global_mutex, as nr_pool will be set to 1 anyway for do_skip_align!
						if (do_skip_align)
						{
							// Rounded translations will be applied to the image upon reading,
							// set the unique translation in the sampling object to the fractional difference
							Matrix1D<DOUBLE> rounded_offset = my_old_offset;
							rounded_offset.selfROUND();
							rounded_offset = my_old_offset - rounded_offset;
							sampling.setOneTranslation(rounded_offset);
						}
		
						// Also set the rotations
						DOUBLE old_rot, old_tilt, old_psi;
						old_rot = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT);
						old_tilt = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT);
						old_psi = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI);
						sampling.setOneOrientation(old_rot, old_tilt, old_psi);
		
					}
					else if (mymodel.orientational_prior_mode != NOPRIOR && iseries == 0)
					{
						// First try if there are some fixed prior angles
						DOUBLE prior_rot = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT_PRIOR);
						DOUBLE prior_tilt = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT_PRIOR);
						DOUBLE prior_psi = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI_PRIOR);
		
						// If there were no defined priors (i.e. their values were 999.), then use the "normal" angles
						if (prior_rot > 998.99 && prior_rot < 999.01)
						{
							prior_rot = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT);
						}
						if (prior_tilt > 998.99 && prior_tilt < 999.01)
						{
							prior_tilt = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT);
						}
						if (prior_psi > 998.99 && prior_psi < 999.01)
						{
							prior_psi = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI);
						}
		
						// For tilted series: convert the angles back onto the untilted ones...
						// Calculate the angles back from the Euler matrix because for tilt series exp_R_mic may have changed them...
						Matrix2D<DOUBLE> A, R_mic(3, 3);
						R_mic(0, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_0);
						R_mic(0, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_1);
						R_mic(0, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_2);
						R_mic(1, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_0);
						R_mic(1, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_1);
						R_mic(1, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_2);
						R_mic(2, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_0);
						R_mic(2, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_1);
						R_mic(2, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_2);
						if (!R_mic.isIdentity())
						{
							Euler_angles2matrix(prior_rot, prior_tilt, prior_psi, A);
							A = R_mic.inv() * A;
							Euler_matrix2angles(A, prior_rot, prior_tilt, prior_psi);
						}
		
						// Select only those orientations that have non-zero prior probability
						sampling.selectOrientationsWithNonZeroPriorProbability(prior_rot, prior_tilt, prior_psi,
																			   sqrt(mymodel.sigma2_rot), sqrt(mymodel.sigma2_tilt), sqrt(mymodel.sigma2_psi));
		
						long int nr_orients = sampling.NrDirections() * sampling.NrPsiSamplings();
						if (nr_orients == 0)
						{
							std::cerr << " sampling.NrDirections()= " << sampling.NrDirections() << " sampling.NrPsiSamplings()= " << sampling.NrPsiSamplings() << std::endl;
							REPORT_ERROR("Zero orientations fall within the local angular search. Increase the sigma-value(s) on the orientations!");
						}
					}
		
					my_old_offset.selfROUND();
		
					exp_old_offset.at(my_image_no) = my_old_offset;
					// Also store priors on translations
					exp_prior.at(my_image_no) = my_prior;
					//Only for copy the offset of each image to GPU by using a large array
					exp_old_offset_H[my_image_no * 2] = XX(my_old_offset);
					exp_old_offset_H[my_image_no * 2 + 1] = YY(my_old_offset);
		
					beamtilt_x_H[my_image_no] = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_BEAMTILT_X);
					beamtilt_y_H[my_image_no] = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_BEAMTILT_Y);
					Cs_H[my_image_no] = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_CS);
					DOUBLE V = 1000. * DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_VOLTAGE);
					lambda_H[my_image_no] = 12.2643247 / sqrt(V * (1. + V * 0.978466e-6));
		
				}
			}
			std::vector< Image<DOUBLE> > local_images, local_rec_images;
			std::vector< MultidimArray<Complex> > local_Fimgs, local_Faux;
			std::vector< MultidimArray<DOUBLE> > local_Fctf_images, local_img_aux;
		
			local_images.clear();
			local_rec_images.clear();
			local_Fimgs.clear();
			local_Faux.clear();
			local_Fctf_images.clear();
			local_img_aux.clear();
		
			local_images.resize(exp_nr_images);
			local_rec_images.resize(exp_nr_images);
			local_Fimgs.resize(exp_nr_images);
			local_Faux.resize(exp_nr_images);
			local_Fctf_images.resize(exp_nr_images);
			local_img_aux.resize(exp_nr_images);
			cudaError cu_error = cudaGetLastError();
			DOUBLE* translated_matrix_H;
			int matrix_size = 3 * 3;
			translated_matrix_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles * matrix_size);
			int image_size = mymodel.ori_size * mymodel.ori_size;
			int nr_red_images = 0;
			if (sub_extract)
				nr_red_images = exp_nr_images;
			cudaMalloc((void**)&image_D, (exp_nr_images+nr_red_images) * image_size * sizeof(DOUBLE));
			cudaMalloc((void**)&rec_image_D, exp_nr_images * image_size * sizeof(DOUBLE));
			cudaMemcpy(image_D, exp_imagedata.data, sizeof(DOUBLE) *image_size * exp_nr_images,  cudaMemcpyHostToDevice);
		
			cu_error = cudaGetLastError();
			if (cu_error != cudaSuccess)
			{
				std::cout << "cudaMemcpy Error :" << __LINE__ << std::endl;
				REPORT_ERROR("cudaMemcpy lambda_D ");
			}
			DOUBLE *image_red_D;
			DOUBLE *rec_image_red_D;
			//sub_extract = false;
			if (sub_extract)
			{
				image_red_D = image_D+(exp_nr_images) * image_size;
				//std::cout << "start sub extraction !!!" << std::endl;
				//cudaMalloc((void**)&image_red_D, exp_nr_images * image_size * sizeof(DOUBLE));
				cudaMalloc((void**)&rec_image_red_D, exp_nr_images * image_size * sizeof(DOUBLE));
				//----------need project!
				cudaMemcpy(image_red_D, image_D, sizeof(DOUBLE) *image_size * exp_nr_images,  cudaMemcpyDeviceToDevice);
				cu_error = cudaGetLastError();
				if (cu_error != cudaSuccess)
				{
					std::cout << "cudaMemcpy Error :" << __LINE__ << std::endl;
					REPORT_ERROR("cudaMemcpy lambda_D ");
				}
				subYellowMK_gpu(image_red_D);
				cu_error = cudaGetLastError();
				if (cu_error != cudaSuccess)
				{
					std::cout << "cudaMemcpy Error :" << __LINE__ << std::endl;
					REPORT_ERROR("subYellowMK_gpu lambda_D ");
				}
			}
			//compare_CPU_GPU(exp_imagedata.data,image_red_D, (exp_nr_images) * image_size, "imagepixels", true);
			cu_error = cudaGetLastError();
			if (cu_error != cudaSuccess)
			{
				std::cout << "cudaMemcpy Error :" << __LINE__ << std::endl;
				REPORT_ERROR("do_yellow_red_mask lambda_D ");
			}
			DOUBLE* normcorr_D;
			cudaMalloc((void**)&normcorr_D, exp_nr_images * sizeof(DOUBLE));
			cudaMemcpy(normcorr_D, normcorr_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
			if (do_norm_correction)
			{
				do_norm_correction_gpu(image_D, normcorr_D, image_size, exp_nr_images,	mymodel.avg_norm_correction);
				if (sub_extract)
				{
					do_norm_correction_gpu(image_red_D, normcorr_D, image_size, exp_nr_images,	mymodel.avg_norm_correction);
				}
			}
		
			cudaError cu_error1 = cudaGetLastError();
			if (cu_error1 != cudaSuccess)
			{
				std::cout << "cudaMemcpy Error :" << __LINE__ << std::endl;
				REPORT_ERROR("cudaMemcpy lambda_D ");
			}
			for (long int ipart = 0; ipart < exp_nr_particles; ipart++)
			{
				long int part_id = exp_ipart_to_part_id[ipart];
		
				// Prevent movies and series at the same time...
				if (mydata.getNrImagesInSeries(part_id) > 1 && do_realign_movies)
				{
					REPORT_ERROR("Not ready yet for dealing with image series at the same time as realigning movie frames....");
				}
		
				for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
				{
					int my_image_no = exp_starting_image_no.at(ipart) + iseries;
					// Which group do I belong?
					//int group_id = mydata.getGroupId(part_id, iseries);
					// Apply the norm_correction term
		
		
					Matrix2D< DOUBLE > tmp;
					translation2DMatrix(exp_old_offset.at(my_image_no), tmp);
		
					tmp = tmp.inv();
					memcpy(translated_matrix_H + ipart * matrix_size, tmp.mdata, sizeof(DOUBLE)* matrix_size);
		
				}
			}
			cudaMalloc((void**)&exp_old_offset_D,  exp_nr_images * 2 * sizeof(DOUBLE));
			cudaMalloc((void**)&translated_matrix_D,  exp_nr_images * matrix_size * sizeof(DOUBLE));
			cudaMemcpy(translated_matrix_D, translated_matrix_H, exp_nr_images * matrix_size * sizeof(DOUBLE), cudaMemcpyHostToDevice);
			cudaMemcpy(exp_old_offset_D, exp_old_offset_H, exp_nr_images * 2 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
		
			selfTranslate_gpu(image_D, translated_matrix_D, exp_old_offset_D, 2, mymodel.ori_size, mymodel.ori_size, image_size, exp_nr_images, DONT_WRAP);
			if (sub_extract)
			{
				selfTranslate_gpu(image_red_D, translated_matrix_D, exp_old_offset_D, 2, mymodel.ori_size, mymodel.ori_size, image_size, exp_nr_images, DONT_WRAP);
			}
			
			if (has_converged && do_use_reconstruct_images)
			{
				cudaMemcpy(rec_image_D, exp_imagedata.data + exp_nr_images * YXSIZE(exp_imagedata), sizeof(DOUBLE) *mymodel.ori_size * mymodel.ori_size * exp_nr_images, cudaMemcpyHostToDevice);
				selfTranslate_gpu(rec_image_D, translated_matrix_D, exp_old_offset_D, 2, mymodel.ori_size, mymodel.ori_size, image_size, exp_nr_images, DONT_WRAP);
				if (sub_extract)
				{
					cudaMemcpy(rec_image_red_D, exp_imagedata.data + exp_nr_images * YXSIZE(exp_imagedata), sizeof(DOUBLE) *mymodel.ori_size * mymodel.ori_size * exp_nr_images, cudaMemcpyHostToDevice);
					selfTranslate_gpu(rec_image_red_D, translated_matrix_D, exp_old_offset_D, 2, mymodel.ori_size, mymodel.ori_size, image_size, exp_nr_images, DONT_WRAP);
				}
			}
		
			free(translated_matrix_H);
			cudaFree(normcorr_D);
			cudaFree(exp_old_offset_D);
			//The GPU codes correspond to the above CXX codes, copy images from exp_imagedata
			long int size_ = 0 ;
			image_size = mymodel.ori_size * mymodel.ori_size;
			size_ = exp_nr_images * image_size;
			//TODO: Apply the norm_correction term
		
			DOUBLE* beamtilt_x_D, *beamtilt_y_D;
			DOUBLE* Cs_D, *lambda_D;
		
		
			cudaMalloc((void**)&beamtilt_x_D, exp_nr_images * sizeof(DOUBLE));
			cudaMalloc((void**)&beamtilt_y_D, exp_nr_images * sizeof(DOUBLE));
			cudaMalloc((void**)&Cs_D, exp_nr_images * sizeof(DOUBLE));
			cudaMalloc((void**)&lambda_D, exp_nr_images * sizeof(DOUBLE));
		
			cu_error = cudaMemcpy(beamtilt_x_D, beamtilt_x_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
			cu_error = cudaMemcpy(beamtilt_y_D, beamtilt_y_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
			cu_error = cudaMemcpy(Cs_D, Cs_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
			cu_error = cudaMemcpy(lambda_D, lambda_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
			if (cu_error != cudaSuccess)
			{
				std::cout << "cudaMemcpy Error :" << __LINE__ << std::endl;
				REPORT_ERROR("cudaMemcpy lambda_D ");
			}
			DOUBLE* raw_data_images, *raw_rec_images;
			raw_data_images = new DOUBLE[size_];
			if (has_converged && do_use_reconstruct_images)
			{
				raw_rec_images = new DOUBLE[size_];
			}
		
			if (cu_error != cudaSuccess)
			{
				std::cout << "cudaMemcpy Error :" << __LINE__ << std::endl;
				REPORT_ERROR("cudaMemcpy lambda_D ");
			}
		
			int dim, xdim, ydim, zdim;
		
			xdim = mymodel.ori_size;
			ydim = mymodel.ori_size;
			zdim  = 1;
			dim = 2 ;
			//Do the CenterFFT for some partiles with GPU
			DOUBLE* local_images_D;
			cudaMalloc((void**)&local_images_D,  size_ * sizeof(DOUBLE));
			cudaMemset(local_images_D, 0, size_ * sizeof(DOUBLE));
			centerFFT_gpu(((has_converged && do_use_reconstruct_images) ? rec_image_D : image_D) , local_images_D, exp_nr_images, dim, xdim,  ydim, zdim, true);
		
			cudaError cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
				exit(EXIT_FAILURE);
			}
			//Foutier Transform for some particles on GPU
			CUFFT_COMPLEX * local_Faux_D;
			cudaDeviceSynchronize();
		
			int ndim = 3;
			if (zdim == 1)
			{
				ndim = 2;
				if (ydim == 1)
				{
					ndim = 1;
				}
			}
			int* N;
			N = (int*)malloc(sizeof(int) * ndim);
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
			cufftHandle fPlanForward_gpu;
#ifdef FLOAT_PRECISION
			cufftResult fftplan1 = cufftPlanMany(&fPlanForward_gpu, ndim, N,
												 NULL, 1, 0,
												 NULL, 1, 0,
												 CUFFT_R2C, exp_nr_images);
			cudaMalloc((void**)&local_Faux_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1)*sizeof(CUFFT_COMPLEX ));
			cufftExecR2C(fPlanForward_gpu, local_images_D, local_Faux_D);
#else
			cufftResult fftplan1 = cufftPlanMany(&fPlanForward_gpu, ndim, N,
												 NULL, 1, 0,
												 NULL, 1, 0,
												 CUFFT_D2Z, exp_nr_images);
			cudaMalloc((void**)&local_Faux_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1)*sizeof(CUFFT_COMPLEX ));
			cufftExecD2Z(fPlanForward_gpu, local_images_D, local_Faux_D);
#endif
			
			ScaleComplexPointwise_gpu(local_Faux_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1), 1.0 / (zdim * ydim * xdim));
		
			int newhdim = mymodel.current_size / 2 + 1;
			int newdim = mymodel.current_size;
			int mem_size;
			if (ndim == 1)
			{
				mem_size = newhdim * 2 * sizeof(DOUBLE);
			}
			else if (ndim == 2)
			{
				mem_size = newdim * newhdim * 2 * sizeof(DOUBLE);
			}
			else if (ndim == 3)
			{
				mem_size = newdim * newdim * newhdim * 2 * sizeof(DOUBLE);
			}
			cudaMalloc((void**)&exp_Fimgs_nomask_D, exp_nr_images * mem_size);
			windowFourierTransform_gpu(local_Faux_D,
									   exp_Fimgs_nomask_D,
									   mymodel.current_size,
									   exp_nr_images,
									   2,
									   (xdim / 2) + 1,
									   ydim,
									   zdim);
		
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
				exit(EXIT_FAILURE);
			}
		
			selfApplyBeamTilt_gpu(exp_Fimgs_nomask_D, beamtilt_x_D, beamtilt_y_D,
								  lambda_D, Cs_D, mymodel.pixel_size, mymodel.ori_size, exp_nr_images,	2,	newhdim,  newdim);
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
				exit(EXIT_FAILURE);
			}
		
			//Do mask on experiment particles
			DOUBLE* Mnoise_D;
			Mnoise_D = NULL;
			softMaskOutsideMap_gpu(image_D, particle_diameter / (2. * mymodel.pixel_size), (DOUBLE)width_mask_edge, Mnoise_D, exp_nr_images, xdim, ydim, zdim);
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
				exit(EXIT_FAILURE);
			}
		
			centerFFT_gpu(image_D , local_images_D, exp_nr_images, dim, xdim,  ydim, zdim, true);
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
				exit(EXIT_FAILURE);
			}
#ifdef FLOAT_PRECISION
			cufftExecR2C(fPlanForward_gpu, local_images_D, local_Faux_D);
#else
			cufftExecD2Z(fPlanForward_gpu, local_images_D, local_Faux_D);
#endif
			ScaleComplexPointwise_gpu(local_Faux_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1), 1.0 / (zdim * ydim * xdim));
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
				exit(EXIT_FAILURE);
			}
			cudaDeviceSynchronize();
			cufftDestroy(fPlanForward_gpu);
			
			DOUBLE* exp_highres_Xi2_imgs_H, *exp_highres_Xi2_imgs_D;
			DOUBLE* exp_power_imgs_H, *exp_power_imgs_D;
			exp_highres_Xi2_imgs_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles);
			exp_power_imgs_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles * (mymodel.ori_size / 2 + 1));
			cudaMalloc((void**)&exp_highres_Xi2_imgs_D, exp_nr_particles * sizeof(DOUBLE));
			cudaMemset(exp_highres_Xi2_imgs_D, 0.0, exp_nr_particles * sizeof(DOUBLE));
			cudaMalloc((void**)&exp_power_imgs_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
			cudaMemset(exp_power_imgs_D, 0.0, exp_nr_particles * (mymodel.ori_size / 2 + 1) * sizeof(DOUBLE));
			calculate_img_power_gpu(local_Faux_D,
									exp_power_imgs_D,
									exp_highres_Xi2_imgs_D,
									exp_nr_images,
									(xdim / 2 + 1),
									ydim,
									zdim,
									mymodel.ori_size,
									mymodel.current_size
								   );
		
			if (mymodel.current_size < mymodel.ori_size)
			{
				cudaMemcpy(exp_power_imgs_H, exp_power_imgs_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);
				cudaMemcpy(exp_highres_Xi2_imgs_H, exp_highres_Xi2_imgs_D, exp_nr_particles * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
				for (long int ipart = 0; ipart < exp_nr_particles; ipart++)
				{
					exp_highres_Xi2_imgs.at(ipart) = exp_highres_Xi2_imgs_H[ipart];
					exp_power_imgs[ipart].resize((mymodel.ori_size / 2 + 1));
					memcpy(exp_power_imgs[ipart].data, exp_power_imgs_H + ipart * (mymodel.ori_size / 2 + 1), (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
				}
			}
			else
			{
				for (long int ipart = 0; ipart < exp_nr_particles; ipart++)
				{
					exp_highres_Xi2_imgs.at(ipart) = 0.;
				}
			}
			cudaFree(exp_power_imgs_D);
			cudaFree(exp_highres_Xi2_imgs_D);
			free(exp_power_imgs_H);
			free(exp_highres_Xi2_imgs_H);
			cudaMalloc((void**)&exp_Fimgs_D, (exp_nr_images+nr_red_images) * mem_size);
			
			cudaMemset(exp_Fimgs_D, 0., (exp_nr_images+nr_red_images) * mem_size);
			windowFourierTransform_gpu(local_Faux_D,
									   exp_Fimgs_D,
									   mymodel.current_size,
									   exp_nr_images,
									   2,
									   (xdim / 2) + 1,
									   ydim,
									   zdim);
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
				exit(EXIT_FAILURE);
			}
		
			//Fourier_image_size = newdim * newhdim;
		
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
				exit(EXIT_FAILURE);
			}
		
			DOUBLE* ctf_related_parameters_H;
			int nr_parameters = 9;
			ctf_related_parameters_H = (DOUBLE*) malloc(nr_parameters * exp_nr_images * sizeof(DOUBLE));
			for (int im = 0; im < exp_nr_images; im++)
			{
				DOUBLE DeltafU = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_DEFOCUS_U);
				DOUBLE DeltafV	= DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_DEFOCUS_V);
				DOUBLE azimuthal_angle	= DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_DEFOCUS_ANGLE);
				DOUBLE kV	= DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_VOLTAGE);
				DOUBLE Cs	= DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_CS);
				DOUBLE Q0	= DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_Q0);
				DOUBLE Bfac = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_BFAC);
				DOUBLE scale = 1.0;
				DOUBLE local_Cs = Cs * 1e7;
				DOUBLE local_kV = kV * 1e3;
				DOUBLE rad_azimuth = DEG2RAD(azimuthal_angle);
		
				DOUBLE defocus_average	 = -(DeltafU + DeltafV) * 0.5;
				DOUBLE defocus_deviation = -(DeltafU - DeltafV) * 0.5;
		
				DOUBLE lambda = 12.2643247 / sqrt(local_kV * (1. + local_kV * 0.978466e-6));
		
				DOUBLE K1 = PI / 2 * 2 * lambda;
				DOUBLE K2 = PI / 2 * local_Cs * lambda * lambda * lambda;
				DOUBLE K3 = sqrt(1 - Q0 * Q0);
				DOUBLE K4 = -Bfac / 4.;
				ctf_related_parameters_H[im * nr_parameters + 0] = defocus_average;
				ctf_related_parameters_H[im * nr_parameters + 1] = defocus_deviation;
				ctf_related_parameters_H[im * nr_parameters + 2] = rad_azimuth;
				ctf_related_parameters_H[im * nr_parameters + 3] = Q0;
				ctf_related_parameters_H[im * nr_parameters + 4] = scale;
				ctf_related_parameters_H[im * nr_parameters + 5] = K1;
				ctf_related_parameters_H[im * nr_parameters + 6] = K2;
				ctf_related_parameters_H[im * nr_parameters + 7] = K3;
				ctf_related_parameters_H[im * nr_parameters + 8] = K4;
		
			}
			cudaMalloc((void**)&exp_Fctf_D, exp_nr_images * newdim * newhdim * sizeof(DOUBLE));
			getFftwImage_gpu(exp_Fctf_D, ctf_related_parameters_H,
							 mymodel.ori_size, mymodel.ori_size, mymodel.pixel_size,
							 ctf_phase_flipped, only_flip_phases, intact_ctf_first_peak, true,
							 exp_nr_images,
							 newhdim,
							 newdim,
							 do_ctf_correction);
		
			free(ctf_related_parameters_H);
			cudaDeviceSynchronize();
			for (long int ipart = 0; ipart < exp_nr_particles; ipart++)
			{
				long int part_id = exp_ipart_to_part_id[ipart];
		
				for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
				{
					int my_image_no = exp_starting_image_no.at(ipart) + iseries;
					// Store Fimg and Fctf
		
					exp_Fimgs[my_image_no].resize(1, newdim, newhdim);
					exp_Fimgs_nomask[my_image_no].resize(1, newdim, newhdim);
		
					exp_Fctfs[my_image_no].resize(1, newdim, newhdim);
				} // end loop iseries
			}// end loop ipart
			transformer.free_memory_gpu();
			transformer.cleanup();
		
			if (sub_extract)
			{
				DOUBLE* Mnoise_D;
				Mnoise_D = NULL;
				softMaskOutsideMap_gpu(image_red_D, particle_diameter / (2. * mymodel.pixel_size), (DOUBLE)width_mask_edge, Mnoise_D, exp_nr_images, xdim, ydim, zdim);
				cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
					exit(EXIT_FAILURE);
				}
				
				DOUBLE* local_images_red_D;
				cudaMalloc((void**)&local_images_red_D,  size_ * sizeof(DOUBLE));
				cudaMemset(local_images_red_D, 0, size_ * sizeof(DOUBLE));
				centerFFT_gpu(image_red_D , local_images_red_D, exp_nr_images, dim, xdim,  ydim, zdim, true);
		
				cudaError cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
					exit(EXIT_FAILURE);
				}
				CUFFT_COMPLEX * local_Faux_red_D;
				
				cudaDeviceSynchronize();
		
				cufftHandle fPlanForward_gpu;
#ifdef FLOAT_PRECISION
				cufftResult fftplan1 = cufftPlanMany(&fPlanForward_gpu, ndim, N,
													 NULL, 1, 0,
													 NULL, 1, 0,
													 CUFFT_R2C, exp_nr_images);
				cudaMalloc((void**)&local_Faux_red_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1)*sizeof(CUFFT_COMPLEX ));
				cufftExecR2C(fPlanForward_gpu, local_images_red_D, local_Faux_red_D);
#else
				cufftResult fftplan1 = cufftPlanMany(&fPlanForward_gpu, ndim, N,
													 NULL, 1, 0,
													 NULL, 1, 0,
													 CUFFT_D2Z, exp_nr_images);
				cudaMalloc((void**)&local_Faux_red_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1)*sizeof(CUFFT_COMPLEX ));
				cufftExecD2Z(fPlanForward_gpu, local_images_red_D, local_Faux_red_D);
#endif
				
				cudaDeviceSynchronize();
				cufftDestroy(fPlanForward_gpu);
		
				ScaleComplexPointwise_gpu(local_Faux_red_D, exp_nr_images * zdim * ydim * (xdim / 2 + 1), 1.0 / (zdim * ydim * xdim));
		
				//DOUBLE* exp_highres_Xi2_imgs_H, *exp_highres_Xi2_imgs_D;
				//DOUBLE* exp_power_imgs_H, *exp_power_imgs_D;
				exp_highres_Xi2_imgs_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles);
				exp_power_imgs_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles * (mymodel.ori_size / 2 + 1));
				cudaMalloc((void**)&exp_highres_Xi2_imgs_D, exp_nr_particles * sizeof(DOUBLE));
				cudaMemset(exp_highres_Xi2_imgs_D, 0.0, exp_nr_particles * sizeof(DOUBLE));
				cudaMalloc((void**)&exp_power_imgs_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
				cudaMemset(exp_power_imgs_D, 0.0, exp_nr_particles * (mymodel.ori_size / 2 + 1) * sizeof(DOUBLE));
				calculate_img_power_gpu(local_Faux_red_D,
										exp_power_imgs_D,
										exp_highres_Xi2_imgs_D,
										exp_nr_images,
										(xdim / 2 + 1),
										ydim,
										zdim,
										mymodel.ori_size,
										mymodel.current_size
									   );
		
				if (mymodel.current_size < mymodel.ori_size)
				{
					cudaMemcpy(exp_power_imgs_H, exp_power_imgs_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);
					cudaMemcpy(exp_highres_Xi2_imgs_H, exp_highres_Xi2_imgs_D, exp_nr_particles * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
					for (long int ipart = 0; ipart < exp_nr_particles; ipart++)
					{
						exp_highres_Xi2_imgs_red.at(ipart) = exp_highres_Xi2_imgs_H[ipart];
						exp_power_imgs_red[ipart].resize((mymodel.ori_size / 2 + 1));
						memcpy(exp_power_imgs_red[ipart].data, exp_power_imgs_H + ipart * (mymodel.ori_size / 2 + 1), (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
					}
				}
				else
				{
					for (long int ipart = 0; ipart < exp_nr_particles; ipart++)
					{
						exp_highres_Xi2_imgs_red.at(ipart) = 0.;
					}
				}
				cudaFree(exp_power_imgs_D);
				cudaFree(exp_highres_Xi2_imgs_D);
				free(exp_power_imgs_H);
				free(exp_highres_Xi2_imgs_H);
			
				//int newhdim = mymodel.current_size/2 + 1;
				//int newdim = mymodel.current_size;
				//int mem_size;
				/*if (ndim == 1)
				{
					mem_size = newhdim * 2 * sizeof(DOUBLE);
				}
				else if (ndim == 2)
				{
					mem_size = newdim * newhdim * 2 * sizeof(DOUBLE);
				}
				else if (ndim == 3)
				{
					mem_size = newdim * newdim * newhdim * 2 * sizeof(DOUBLE);
				}*/
				CUFFT_COMPLEX *exp_Fimgs_red_D ;
				exp_Fimgs_red_D = exp_Fimgs_D + exp_nr_images * (mem_size /sizeof(CUFFT_COMPLEX));
				//cudaMalloc((void**)&exp_Fimgs_red_D, exp_nr_images * mem_size);
				//cudaMemset(exp_Fimgs_red_D, 0., exp_nr_images * mem_size);
				
				/*Complex *local_Faux_red_H;
				local_Faux_red_H = (Complex *) malloc(exp_nr_images  * newdim * newhdim*sizeof(CUFFT_COMPLEX ));
				cudaMemcpy(local_Faux_red_H, local_Faux_red_D, sizeof(CUFFT_COMPLEX) *exp_nr_images   * newdim * newhdim, cudaMemcpyDeviceToHost);
				compare_CPU_GPU(local_Faux_red_H, local_Faux_D, exp_nr_images  * newdim * newhdim, "image_red_D", true);
		
				free(local_Faux_red_H);
				std::cout<< "before compare current size " << mymodel.current_size << " mem_size  " <<mem_size<< std::endl;*/
				/*windowFourierTransform_gpu(local_Faux_D,
									   exp_Fimgs_D,
									   mymodel.current_size,
									   exp_nr_images,
									   2,
									   (xdim/2) + 1,
									   ydim,
									   zdim);*/
				
				windowFourierTransform_gpu(local_Faux_red_D,
												   exp_Fimgs_red_D,
												   mymodel.current_size,
												   exp_nr_images,
												   2,
												   (xdim/2) + 1,
												   ydim,
												   zdim);
				
				cudaDeviceSynchronize();
				cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
					exit(EXIT_FAILURE);
				}
		
				//std::cout<< "current size " << mymodel.current_size << " xdim  " <<(exp_nr_images * newdim * newhdim*sizeof(CUFFT_COMPLEX ))<< "	newdim	" <<  exp_nr_images*mem_size << "  newh dim" << newhdim<<  std::endl;
				/*Complex *image_red_H;
				image_red_H = (Complex *) malloc(exp_nr_images	* newdim * newhdim*sizeof(CUFFT_COMPLEX ));
				cudaMemcpy(image_red_H, exp_Fimgs_red_D, (exp_nr_images  * newdim * newhdim*sizeof(CUFFT_COMPLEX )), cudaMemcpyDeviceToHost);
				//cudaMemcpy(data_gpu_H, data_gpu, size * sizeof(Complex), cudaMemcpyDeviceToHost);
				cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
					exit(EXIT_FAILURE);
				}
				compare_CPU_GPU(image_red_H, exp_Fimgs_D, exp_nr_images * newdim * newhdim, "image_red_D", true);
		
				std::cout<< "current size " << mymodel.current_size << " xdim  " << (xdim/2)+1<< std::endl;
				
				free(image_red_H);*/
				cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
					exit(EXIT_FAILURE);
				}
				//cudaFree(image_red_D);
				cudaFree(rec_image_red_D);
				cudaFree(local_images_red_D);
				cudaFree(local_Faux_red_D);
				
		
			}
		
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
				exit(EXIT_FAILURE);
			}
			
			cudaFree(translated_matrix_D);
			cudaFree(beamtilt_x_D);
			cudaFree(beamtilt_y_D);
			cudaFree(Cs_D);
			cudaFree(lambda_D);
			cudaFree(image_D);
			cudaFree(rec_image_D);
			cudaFree(local_images_D);
			cudaFree(local_Faux_D);
		
			delete [] normcorr_H;
			delete [] beamtilt_x_H;
			delete [] beamtilt_y_H;
			delete [] Cs_H;
			delete [] lambda_H;
			delete []exp_old_offset_H;
			delete [] raw_data_images;
			free(N);
			if (has_converged && do_use_reconstruct_images)
			{
				delete []raw_rec_images;
			}
			cudaStat = cudaGetLastError();
			if (cudaStat != cudaSuccess)
			{
				printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
				exit(EXIT_FAILURE);
			}
		
		}


void MlOptimiser::getAllSquaredDifferences_gpu()
{

#ifdef TIMING
	if (exp_ipass == 0)
	{
		timer.tic(TIMING_ESP_DIFF1);
	}
	else
	{
		timer.tic(TIMING_ESP_DIFF2);
	}
#endif



	//#define DEBUG_GETALLDIFF2
#ifdef DEBUG_GETALLDIFF2
	std::cerr << " ipass= " << exp_ipass << " exp_current_oversampling= " << exp_current_oversampling << std::endl;
	std::cerr << " sampling.NrPsiSamplings(exp_current_oversampling)= " << sampling.NrPsiSamplings(exp_current_oversampling) << std::endl;
	std::cerr << " sampling.NrTranslationalSamplings(exp_current_oversampling)= " << sampling.NrTranslationalSamplings(exp_current_oversampling) << std::endl;
	std::cerr << " sampling.NrSamplingPoints(exp_current_oversampling)= " << sampling.NrSamplingPoints(exp_current_oversampling) << std::endl;
	std::cerr << " sampling.oversamplingFactorOrientations(exp_current_oversampling)= " << sampling.oversamplingFactorOrientations(exp_current_oversampling) << std::endl;
	std::cerr << " sampling.oversamplingFactorTranslations(exp_current_oversampling)= " << sampling.oversamplingFactorTranslations(exp_current_oversampling) << std::endl;
#endif
	DOUBLE inti_diff = -999.;
	int Weight_size_per_class = exp_nr_rot * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans;
	exp_Mweight_D_size = mymodel.nr_classes * Weight_size_per_class;
	cudaMalloc((void**)&exp_Mweight_D,  exp_nr_particles * mymodel.nr_classes * Weight_size_per_class* sizeof(DOUBLE));
	init_exp_mweight_gpu(exp_Mweight_D, inti_diff, exp_nr_particles * mymodel.nr_classes * Weight_size_per_class);
	if (exp_ipass == 0)
	{
		exp_Mcoarse_significant.clear();
	}
	cudaMalloc((void**)&exp_min_diff2_D, exp_nr_particles * sizeof(DOUBLE));
	exp_min_diff2.clear();
	exp_min_diff2.resize(exp_nr_particles);
	init_exp_min_diff2_gpu(exp_min_diff2_D, 99e99, exp_nr_particles);

	// Use pre-sized vectors instead of push_backs!!
	exp_local_Fimgs_shifted.clear();
	exp_local_Fimgs_shifted.resize(exp_nr_images * sampling.NrTranslationalSamplings(exp_current_oversampling));
	exp_local_Fimgs_shifted_nomask.clear();
	exp_local_Fimgs_shifted_nomask.resize(exp_nr_images * sampling.NrTranslationalSamplings(exp_current_oversampling));
	exp_local_Minvsigma2s.clear();
	exp_local_Minvsigma2s.resize(exp_nr_images);
	exp_local_Fctfs.clear();
	exp_local_Fctfs.resize(exp_nr_images);
	exp_local_sqrtXi2.clear();
	exp_local_sqrtXi2.resize(exp_nr_images);

	// copy the data to GPU for the following two smapling
	int input_size = exp_Fimgs[0].zyxdim;
	DOUBLE* local_exp_Fimgs, *local_exp_Fimgs_nomask, *local_Fctf_H;
	local_exp_Fimgs = (DOUBLE*) malloc(input_size * 2 * sizeof(DOUBLE) * exp_nr_images);
	local_exp_Fimgs_nomask = (DOUBLE*) malloc(input_size * 2 * sizeof(DOUBLE) * exp_nr_images);
	local_Fctf_H = (DOUBLE*) malloc(exp_Fctfs[0].zyxdim * sizeof(DOUBLE) * exp_nr_images);


	//Prepare the maxtrix A and its invs

	// TODO: MAKE SURE THAT ALL PARTICLES IN SomeParticles ARE FROM THE SAME AREA, SO THAT THE R_mic CAN BE RE_USED!!!
	//for (exp_iseries = 0; exp_iseries < mydata.getNrImagesInSeries(part_id); exp_iseries++)
	for (exp_iseries = 0; exp_iseries < mydata.getNrImagesInSeries((mydata.ori_particles[exp_my_first_ori_particle]).particles_id[0]); exp_iseries++)
	{

		// Get all shifted versions of the (downsized) images, their (downsized) CTFs and their inverted Sigma2 matrices
		doThreadPrecalculateShiftedImagesCtfsAndInvSigma2s_gpu();
		// Get micrograph transformation matrix. Note that for all pooled particles (exp_my_first_particle-exp_my_last_particle)
		// the same exp_R_mic will be used in order to re-use the reference projections
		// This is the reason why all pooled particles should come from the same micrograph
		// TODO: THAT STILL NEEDS TO BE CONFIRMED!!!! CURRENTLY NO CHECK ON SAME-PARTICLENAME IN EACH POOL!!!
		// WORKAROUND FOR NOW: just set --pool 1
		//exp_R_mic = mydata.getMicrographTransformationMatrix((mydata.ori_particles[exp_my_first_ori_particle]).particles_id[0], exp_iseries);
		int my_image_no = exp_starting_image_no[0] + exp_iseries;
		// Get micrograph transformation matrix
		exp_R_mic.resize(3, 3);
		exp_R_mic(0, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_0);
		exp_R_mic(0, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_1);
		exp_R_mic(0, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_2);
		exp_R_mic(1, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_0);
		exp_R_mic(1, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_1);
		exp_R_mic(1, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_2);
		exp_R_mic(2, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_0);
		exp_R_mic(2, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_1);
		exp_R_mic(2, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_2);

		cudaError cudaStat = cudaGetLastError();
		// Loop from iclass_min to iclass_max to deal with seed generation in first iteration
		/*for (exp_iclass = iclass_min; exp_iclass <= iclass_max; exp_iclass++)
		{
			if (mymodel.pdf_class[exp_iclass] > 0.)
			{
				// GPU function
				doThreadGetSquaredDifferencesAllOrientations_gpu();

				cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_minimal_weight_per_particle_gpu returned error code %d, line(%d), %s, %d\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat), exp_ipass);
					exit(EXIT_FAILURE);
				}
			} // end if mymodel.pdf_class[iclass] > 0.
		} // end loop iclass*/
		doThreadGetSquaredDifferencesAllOrientations_multi_class_gpu();
		//=======================================================
		//Getting the minmal weight for each particle using the thrust
		if (cudaStat != cudaSuccess)
		{
			printf("kernel calculate_minimal_weight_per_particle_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}
		DOUBLE mini_weight_particle[exp_nr_images];
		calculate_minimal_weight_per_particle_gpu(exp_Mweight_D + iclass_min * Weight_size_per_class,
		                                          exp_min_diff2_D,
		                                          inti_diff,
		                                          (iclass_max - iclass_min + 1) * Weight_size_per_class ,
		                                          exp_nr_particles,
		                                          exp_Mweight_D_size);
		cudaStat = cudaGetLastError();

		if (cudaStat != cudaSuccess)
		{
			printf("kernel calculate_minimal_weight_per_particle_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}
		cudaMemcpy(mini_weight_particle, exp_min_diff2_D, sizeof(DOUBLE) * exp_nr_images, cudaMemcpyDeviceToHost);
		for (int i = 0; i < exp_nr_particles; i++)
		{
			exp_min_diff2[i] = mini_weight_particle[i];
                        if (mini_weight_particle[i] > 1e17) printf("exp_min_diff2 = %f exp_ipass = %d\n", exp_min_diff2[i], exp_ipass);
			//std::cout << "  " << exp_min_diff2[i] << "  " << (iclass_max - iclass_min + 1) * Weight_size_per_class;
                        //if (exp_min_diff2[i] > 1e17) { printf("min_diff = %f ", exp_min_diff2[i]);DOUBLE tmp[24192]; cudaMemcpy(tmp, exp_Mweight_D + i*exp_Mweight_D_size, 24192*sizeof(DOUBLE), cudaMemcpyDeviceToHost); for (int j = 0; j < 24192; j++) printf("%f ", tmp[j]); printf("\n"); }


		}
			//std::cout << "  " << inti_diff  << "  " << exp_ipass <<  "  iter = " << iter << std::endl;

	} // end loop iseries
	free(local_exp_Fimgs) ;
	free(local_exp_Fimgs_nomask);
	free(local_Fctf_H);
	if (exp_ipass == 1)
	{
		cudaFree(exp_Fimgs_D);
		cudaFree(exp_Fimgs_nomask_D);
		cudaFree(exp_Fctf_D);
	}


#ifdef DEBUG_GETALLDIFF2b
	for (long int part_id = exp_my_first_particle, ipart = 0; part_id <= exp_my_last_particle; part_id++, ipart++)
	{
		if (exp_min_diff2[ipart] < 0.)
		{
			std::cerr << "Negative min_diff2...." << std::endl;
			std::cerr << " ipart= " << ipart << " part_id= " << part_id << std::endl;
			std::cerr << " do_firstiter_cc= " << do_firstiter_cc << std::endl;
			int group_id = mydata.getGroupId(part_id, 0);
			std::cerr << " group_id= " << group_id << std::endl;
			std::cerr << " ml_model.sigma2_noise[group_id]= " << mymodel.sigma2_noise[group_id] << std::endl;
		}
	}
#endif

#ifdef TIMING
	if (exp_ipass == 0)
	{
		timer.toc(TIMING_ESP_DIFF1);
	}
	else
	{
		timer.toc(TIMING_ESP_DIFF2);
	}
#endif

}

void MlOptimiser::doThreadPrecalculateShiftedImagesCtfsAndInvSigma2s_gpu()
/*{
	int in_size = exp_Fimgs[0].zyxdim;
	CUFFT_COMPLEX * local_Fimgs_D, *local_Fimg_nomask_D;
	DOUBLE* exp_local_sqrtXi2_H;
	DOUBLE* local_Faux_H;
	int* group_id_H, *group_id_D;
	int window_img_size, size_fft;
	int* local_myMresol, *local_myMresol_D;
	DOUBLE* sigma2_noise_H, *sigma2_noise_D;
	DOUBLE* oversampled_translations_H;

	int ndim  = exp_Fimgs[0].getDim();
	int newhdim = exp_current_image_size / 2 + 1;
	int newdim = exp_current_image_size;
	int xdim = mymodel.current_size / 2 + 1;
	int ydim = mymodel.current_size;
	int zdim = 1;
	int image_size_Fctf = exp_Fctfs[0].zyxdim;
	if (ndim == 1)
	{
		window_img_size = newhdim;
		shift_img_x = newhdim;
		shift_img_size = newhdim;

	}
	else if (ndim == 2)
	{
		window_img_size = newdim * newhdim;
		shift_img_x = newhdim;
		shift_img_y = newdim;
		shift_img_size = shift_img_y * shift_img_x;
	}
	else if (ndim == 3)
	{
		window_img_size = newdim * newdim * newhdim;
		shift_img_x = newhdim;
		shift_img_y = newdim;
		shift_img_z = newdim;
		shift_img_size = shift_img_z * shift_img_y * shift_img_x;
	}
	cudaMalloc((void**)&local_Fimgs_D, exp_nr_images * window_img_size * sizeof(CUFFT_COMPLEX ));
	cudaMalloc((void**)&local_Fimg_nomask_D, exp_nr_images * window_img_size * sizeof(CUFFT_COMPLEX ));
	//windows the fourier Transform
	windowFourierTransform_gpu(exp_Fimgs_D,
	                           local_Fimgs_D,
	                           exp_current_image_size,
	                           exp_nr_images,
	                           2,
	                           xdim,
	                           ydim,
	                           zdim);

	// Also precalculate the sqrt of the sum of all Xi2

	cudaMalloc((void**)&exp_local_sqrtXi2_D, exp_nr_images * sizeof(DOUBLE));
	exp_local_sqrtXi2_H = (DOUBLE*) malloc(exp_nr_images * sizeof(DOUBLE));
	if ((iter == 1 && do_firstiter_cc) || do_always_cc)
	{
		calculate_local_sqrtXi2_gpu(local_Fimgs_D, exp_local_sqrtXi2_D, exp_nr_images,  window_img_size);

		cudaMemcpy(exp_local_sqrtXi2_H, exp_local_sqrtXi2_D, exp_nr_images * sizeof(DOUBLE), cudaMemcpyDeviceToHost);

		for (int i = 0; i < exp_nr_images; i++)
		{
			exp_local_sqrtXi2[i] = exp_local_sqrtXi2_H[i];

		}
	}


	oversampled_translations_H = (DOUBLE*) malloc(exp_nr_trans * exp_nr_oversampled_trans * ndim * sizeof(DOUBLE));
	// Store all translated variants of Fimg
	for (long int ipart = 0;  ipart < exp_nr_images; ipart++)
	{

		long int part_id = exp_ipart_to_part_id[ipart];

		int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
		int my_trans_image = my_image_no * exp_nr_trans * exp_nr_oversampled_trans;
		for (long int itrans = 0; itrans < exp_nr_trans; itrans++)
		{
			// First get the non-oversampled translations as defined by the sampling object
			std::vector<Matrix1D <DOUBLE> > oversampled_translations;
			sampling.getTranslations(itrans, exp_current_oversampling, oversampled_translations);

			// Then loop over all its oversampled relatives
			for (long int iover_trans = 0; iover_trans < exp_nr_oversampled_trans; iover_trans++)
			{
				for (int i = 0 ; i < ndim; i++)
				{
					oversampled_translations_H[(itrans * exp_nr_oversampled_trans + iover_trans)*ndim + i] = oversampled_translations[iover_trans].vdata[i];
				}
				my_trans_image++;
			}
		}

	}

	cudaMalloc((void**)&exp_local_Fimgs_shifted_D, exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * newhdim * newdim * zdim * 2 * sizeof(DOUBLE));

	shiftImageInFourierTransform_gpu(local_Fimgs_D,
	                                 exp_local_Fimgs_shifted_D,
	                                 (DOUBLE) mymodel.ori_size, oversampled_translations_H,
	                                 exp_nr_images, exp_nr_trans, exp_nr_oversampled_trans,
	                                 newhdim, newdim, zdim);

	int nr_sampling_passes = (adaptive_oversampling > 0) ? 2 : 1;
	if (exp_ipass == nr_sampling_passes - 1)
	{
		cudaMalloc((void**)&exp_local_Fimgs_shifted_nomask_D, exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * newhdim * newdim * zdim * 2 * sizeof(DOUBLE));
		windowFourierTransform_gpu(exp_Fimgs_nomask_D,
		                           local_Fimg_nomask_D,
		                           exp_current_image_size,
		                           exp_nr_images,
		                           2,
		                           xdim,
		                           ydim,
		                           zdim);
		shiftImageInFourierTransform_gpu(local_Fimg_nomask_D,
		                                 exp_local_Fimgs_shifted_nomask_D,
		                                 (DOUBLE) mymodel.ori_size, oversampled_translations_H,
		                                 exp_nr_images, exp_nr_trans, exp_nr_oversampled_trans,
		                                 newhdim, newdim, zdim);



	}
	cudaDeviceSynchronize();

	cudaMalloc((void**)&exp_local_Fctfs_D, (exp_nr_images * window_img_size * sizeof(DOUBLE)));
	windowFourierTransform_gpu(exp_Fctf_D,
	                           exp_local_Fctfs_D,
	                           exp_current_image_size,
	                           exp_nr_images,
	                           2,
	                           exp_Fctfs[0].xdim,
	                           exp_Fctfs[0].ydim,
	                           zdim);

	size_fft = exp_nr_images * window_img_size;
	local_Faux_H = (DOUBLE*) malloc(sizeof(DOUBLE) * size_fft);
	cudaMemcpy(local_Faux_H, exp_local_Fctfs_D, size_fft * sizeof(DOUBLE), cudaMemcpyDeviceToHost);

	//copy the ctf data back to CPU if necessory
	for (long int ipart = 0;  ipart < exp_nr_images; ipart++)
	{
		long int part_id = exp_ipart_to_part_id[ipart];
		int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
		//MultidimArray<DOUBLE> Fctf;
		exp_local_Fctfs[my_image_no].resize(newdim, newhdim);
		memcpy(exp_local_Fctfs[my_image_no].data, local_Faux_H + my_image_no * window_img_size,  window_img_size * sizeof(DOUBLE));
	}
	free(local_Faux_H);

	group_id_H = (int*)malloc(exp_nr_images * sizeof(int));
	cudaMalloc((void**)&group_id_D, exp_nr_images * sizeof(int));
	// Get micrograph id (for choosing the right sigma2_noise)
	for (long int ipart = 0;  ipart < exp_nr_images; ipart++)
	{
		long int part_id = exp_ipart_to_part_id[ipart];
		int group_id = mydata.getGroupId(part_id, exp_iseries);
		group_id_H[ipart] = group_id;

	}

	//Calculate the  Minvsigma2 with GPU
	int myMresol_size = ((newdim == coarse_size) ? Mresol_coarse.zyxdim : Mresol_fine.zyxdim);
	int image_size = zdim * newdim * newhdim;
	local_myMresol = (int*) malloc(myMresol_size * sizeof(int));
	sigma2_noise_H = (DOUBLE*)malloc(mydata.numberOfGroups() * mymodel.sigma2_noise[0].zyxdim * sizeof(DOUBLE));

	for (int i = 0 ; i < myMresol_size; i++)
	{
		local_myMresol[i] = (newdim == coarse_size) ? Mresol_coarse.data[i] : Mresol_fine.data[i];
	}

	for (int k = 0; k < mydata.numberOfGroups(); k++)
	{
		for (int i = 0 ; i < mymodel.sigma2_noise[0].zyxdim; i++)
		{
			sigma2_noise_H[k * mymodel.sigma2_noise[0].zyxdim + i] =  DIRECT_A1D_ELEM(mymodel.sigma2_noise[k], i);
		}

	}
	cudaMalloc((void**)&local_myMresol_D, myMresol_size * sizeof(int));
	cudaMalloc((void**)&sigma2_noise_D, mydata.numberOfGroups()*mymodel.sigma2_noise[0].zyxdim * sizeof(DOUBLE));
	cudaMemcpy(local_myMresol_D, local_myMresol, myMresol_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(sigma2_noise_D, sigma2_noise_H, mydata.numberOfGroups()*mymodel.sigma2_noise[0].zyxdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cudaMemcpy(group_id_D, group_id_H, exp_nr_images * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&exp_Minvsigma2s_D, image_size * exp_nr_images * sizeof(DOUBLE));

	cudaMemset(exp_Minvsigma2s_D, 0., image_size * exp_nr_images * sizeof(DOUBLE));
	calculate_Minvsigma2_gpu(exp_Minvsigma2s_D,  local_myMresol_D, sigma2_noise_D, group_id_D,   sigma2_fudge, exp_nr_images,  zdim * newdim * newhdim, myMresol_size, mymodel.sigma2_noise[0].zyxdim);

	size_fft = exp_nr_images * image_size;
	local_Faux_H = (DOUBLE*) malloc(sizeof(DOUBLE) * size_fft);
	cudaMemcpy(local_Faux_H, exp_Minvsigma2s_D, size_fft * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	for (long int ipart = 0;  ipart < exp_nr_images; ipart++)
	{
		int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
		exp_local_Minvsigma2s[my_image_no].resize(zdim, newdim, newhdim);
		memcpy(exp_local_Minvsigma2s[my_image_no].data, local_Faux_H + my_image_no * image_size, image_size * sizeof(DOUBLE));

	}
#ifdef TIMING
	timer.toc(TIMING_DIFF_SHIFT);
#endif

	cudaFree(local_Fimgs_D);
	cudaFree(local_Fimg_nomask_D);
	cudaFree(local_myMresol_D);
	cudaFree(sigma2_noise_D);
	cudaFree(group_id_D);

	free(exp_local_sqrtXi2_H);
	free(oversampled_translations_H);
	free(group_id_H);
	free(sigma2_noise_H);
	free(local_myMresol);
	free(local_Faux_H);


}
*/
	{
		//sub_extract = false;
		int in_size = exp_Fimgs[0].zyxdim;
		CUFFT_COMPLEX * local_Fimgs_D, *local_Fimg_nomask_D;
		DOUBLE* exp_local_sqrtXi2_H;
		DOUBLE* local_Faux_H;
		int* group_id_H, *group_id_D;
		int window_img_size, size_fft;
		int* local_myMresol, *local_myMresol_D;
		DOUBLE* sigma2_noise_H, *sigma2_noise_D;
		DOUBLE* oversampled_translations_H;
	
		int ndim  = exp_Fimgs[0].getDim();
		int newhdim = exp_current_image_size / 2 + 1;
		int newdim = exp_current_image_size;
		int xdim = mymodel.current_size / 2 + 1;
		int ydim = mymodel.current_size;
		int zdim = 1;
		int image_size_Fctf = exp_Fctfs[0].zyxdim;
		if (ndim == 1)
		{
			window_img_size = newhdim;
			shift_img_x = newhdim;
			shift_img_size = newhdim;
	
		}
		else if (ndim == 2)
		{
			window_img_size = newdim * newhdim;
			shift_img_x = newhdim;
			shift_img_y = newdim;
			shift_img_size = shift_img_y * shift_img_x;
		}
		else if (ndim == 3)
		{
			window_img_size = newdim * newdim * newhdim;
			shift_img_x = newhdim;
			shift_img_y = newdim;
			shift_img_z = newdim;
			shift_img_size = shift_img_z * shift_img_y * shift_img_x;
		}
		cudaMalloc((void**)&local_Fimgs_D, exp_nr_images * window_img_size * sizeof(CUFFT_COMPLEX ));
		cudaMalloc((void**)&local_Fimg_nomask_D, exp_nr_images * window_img_size * sizeof(CUFFT_COMPLEX ));
		//windows the fourier Transform
		cudaError cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel shiftImageInFourierTransform_gpu subMK ");
		}
		windowFourierTransform_gpu(exp_Fimgs_D,
								   local_Fimgs_D,
								   exp_current_image_size,
								   exp_nr_images,
								   2,
								   xdim,
								   ydim,
								   zdim);
	
		
		cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel shiftImageInFourierTransform_gpu subMK ");
		}
		oversampled_translations_H = (DOUBLE*) malloc(exp_nr_trans * exp_nr_oversampled_trans * ndim * sizeof(DOUBLE));
		// Store all translated variants of Fimg
		for (long int ipart = 0;  ipart < exp_nr_images; ipart++)
		{
	
			long int part_id = exp_ipart_to_part_id[ipart];
	
			int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
			int my_trans_image = my_image_no * exp_nr_trans * exp_nr_oversampled_trans;
			for (long int itrans = 0; itrans < exp_nr_trans; itrans++)
			{
				// First get the non-oversampled translations as defined by the sampling object
				std::vector<Matrix1D <DOUBLE> > oversampled_translations;
				sampling.getTranslations(itrans, exp_current_oversampling, oversampled_translations);
	
				// Then loop over all its oversampled relatives
				for (long int iover_trans = 0; iover_trans < exp_nr_oversampled_trans; iover_trans++)
				{
					for (int i = 0 ; i < ndim; i++)
					{
						oversampled_translations_H[(itrans * exp_nr_oversampled_trans + iover_trans)*ndim + i] = oversampled_translations[iover_trans].vdata[i];
					}
					my_trans_image++;
				}
			}
	
		}
		long int  red_shift_images_size = (sub_extract) ? ( exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * newhdim * newdim * zdim) : 0;
		cudaMalloc((void**)&exp_local_Fimgs_shifted_D, (red_shift_images_size+exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * newhdim * newdim * zdim) * 2 * sizeof(DOUBLE));
		   cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel shiftImageInFourierTransform_gpu subMK ");
		}
		shiftImageInFourierTransform_gpu(local_Fimgs_D,
										 exp_local_Fimgs_shifted_D,
										 (DOUBLE) mymodel.ori_size, oversampled_translations_H,
										 exp_nr_images, exp_nr_trans, exp_nr_oversampled_trans,
										 newhdim, newdim, zdim);
		//Doing the shift for sub_masked images
		cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel shiftImageInFourierTransform_gpu subMK ");
		}
		if(sub_extract)
		{
			CUFFT_COMPLEX *exp_Fimgs_red_D = exp_Fimgs_D +in_size*exp_nr_images;
			CUFFT_COMPLEX *exp_local_Fimgs_shifted_red_D = exp_local_Fimgs_shifted_D +red_shift_images_size;
			//cudaMalloc((void**)&exp_local_Fimgs_shifted_red_D, exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * newhdim * newdim * zdim * 2 * sizeof(DOUBLE));
			windowFourierTransform_gpu(exp_Fimgs_red_D,
								   local_Fimgs_D,
								   exp_current_image_size,
								   exp_nr_images,
								   2,
								   xdim,
								   ydim,
								   zdim);
			
			shiftImageInFourierTransform_gpu(local_Fimgs_D,
										 exp_local_Fimgs_shifted_red_D,
										 (DOUBLE) mymodel.ori_size, oversampled_translations_H,
										 exp_nr_images, exp_nr_trans, exp_nr_oversampled_trans,
										 newhdim, newdim, zdim);
			
			cudaDeviceSynchronize();
			
			cu_error = cudaGetLastError();
			if (cu_error != cudaSuccess)
			{
				std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
				REPORT_ERROR("cudaKernel get2DFourierTransform_gpu subMK ");
			}
	
		}
		
		// Also precalculate the sqrt of the sum of all Xi2
		
		cudaMalloc((void**)&exp_local_sqrtXi2_D, exp_nr_images * sizeof(DOUBLE));
		exp_local_sqrtXi2_H = (DOUBLE*) malloc(exp_nr_images * sizeof(DOUBLE));
		if ((iter == 1 && do_firstiter_cc) || do_always_cc)
		{
			calculate_local_sqrtXi2_gpu(local_Fimgs_D, exp_local_sqrtXi2_D, exp_nr_images,	window_img_size);
	
			cudaMemcpy(exp_local_sqrtXi2_H, exp_local_sqrtXi2_D, exp_nr_images * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	
			for (int i = 0; i < exp_nr_images; i++)
			{
				exp_local_sqrtXi2[i] = exp_local_sqrtXi2_H[i];
	
			}
		}
	
		int nr_sampling_passes = (adaptive_oversampling > 0) ? 2 : 1;
		if (exp_ipass == nr_sampling_passes - 1)
		{
			cudaMalloc((void**)&exp_local_Fimgs_shifted_nomask_D, exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * newhdim * newdim * zdim * 2 * sizeof(DOUBLE));
			windowFourierTransform_gpu(exp_Fimgs_nomask_D,
											   local_Fimg_nomask_D,
											   exp_current_image_size,
											   exp_nr_images,
											   2,
											   xdim,
											   ydim,
											   zdim);
			
			shiftImageInFourierTransform_gpu(local_Fimg_nomask_D,
											 exp_local_Fimgs_shifted_nomask_D,
											 (DOUBLE) mymodel.ori_size, oversampled_translations_H,
											 exp_nr_images, exp_nr_trans, exp_nr_oversampled_trans,
											 newhdim, newdim, zdim);
	
	
	
		}
		cudaDeviceSynchronize();
	
		cudaMalloc((void**)&exp_local_Fctfs_D, (exp_nr_images * window_img_size * sizeof(DOUBLE)));
		windowFourierTransform_gpu(exp_Fctf_D,
								   exp_local_Fctfs_D,
								   exp_current_image_size,
								   exp_nr_images,
								   2,
								   exp_Fctfs[0].xdim,
								   exp_Fctfs[0].ydim,
								   zdim);
	
		size_fft = exp_nr_images * window_img_size;
		local_Faux_H = (DOUBLE*) malloc(sizeof(DOUBLE) * size_fft);
		cudaMemcpy(local_Faux_H, exp_local_Fctfs_D, size_fft * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	
		//copy the ctf data back to CPU if necessory
		for (long int ipart = 0;  ipart < exp_nr_images; ipart++)
		{
			long int part_id = exp_ipart_to_part_id[ipart];
			int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
			//MultidimArray<DOUBLE> Fctf;
			exp_local_Fctfs[my_image_no].resize(newdim, newhdim);
			memcpy(exp_local_Fctfs[my_image_no].data, local_Faux_H + my_image_no * window_img_size,  window_img_size * sizeof(DOUBLE));
		}
		free(local_Faux_H);
	
		group_id_H = (int*)malloc(exp_nr_images * sizeof(int));
		cudaMalloc((void**)&group_id_D, exp_nr_images * sizeof(int));
		// Get micrograph id (for choosing the right sigma2_noise)
		for (long int ipart = 0;  ipart < exp_nr_images; ipart++)
		{
			long int part_id = exp_ipart_to_part_id[ipart];
			int group_id = mydata.getGroupId(part_id, exp_iseries);
			group_id_H[ipart] = group_id;
	
		}
	
		//Calculate the  Minvsigma2 with GPU
		int myMresol_size = ((newdim == coarse_size) ? Mresol_coarse.zyxdim : Mresol_fine.zyxdim);
		int image_size = zdim * newdim * newhdim;
		local_myMresol = (int*) malloc(myMresol_size * sizeof(int));
		sigma2_noise_H = (DOUBLE*)malloc(mydata.numberOfGroups() * mymodel.sigma2_noise[0].zyxdim * sizeof(DOUBLE));
	
		for (int i = 0 ; i < myMresol_size; i++)
		{
			local_myMresol[i] = (newdim == coarse_size) ? Mresol_coarse.data[i] : Mresol_fine.data[i];
		}
	
		for (int k = 0; k < mydata.numberOfGroups(); k++)
		{
			for (int i = 0 ; i < mymodel.sigma2_noise[0].zyxdim; i++)
			{
				sigma2_noise_H[k * mymodel.sigma2_noise[0].zyxdim + i] =  DIRECT_A1D_ELEM(mymodel.sigma2_noise[k], i);
			}
	
		}
		cudaMalloc((void**)&local_myMresol_D, myMresol_size * sizeof(int));
		cudaMalloc((void**)&sigma2_noise_D, mydata.numberOfGroups()*mymodel.sigma2_noise[0].zyxdim * sizeof(DOUBLE));
		cudaMemcpy(local_myMresol_D, local_myMresol, myMresol_size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(sigma2_noise_D, sigma2_noise_H, mydata.numberOfGroups()*mymodel.sigma2_noise[0].zyxdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
		cudaMemcpy(group_id_D, group_id_H, exp_nr_images * sizeof(int), cudaMemcpyHostToDevice);
	
		cudaMalloc((void**)&exp_Minvsigma2s_D, image_size * exp_nr_images * sizeof(DOUBLE));
	
		cudaMemset(exp_Minvsigma2s_D, 0., image_size * exp_nr_images * sizeof(DOUBLE));
		calculate_Minvsigma2_gpu(exp_Minvsigma2s_D,  local_myMresol_D, sigma2_noise_D, group_id_D,	 sigma2_fudge, exp_nr_images,  zdim * newdim * newhdim, myMresol_size, mymodel.sigma2_noise[0].zyxdim);
	
		size_fft = exp_nr_images * image_size;
		local_Faux_H = (DOUBLE*) malloc(sizeof(DOUBLE) * size_fft);
		cudaMemcpy(local_Faux_H, exp_Minvsigma2s_D, size_fft * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
		for (long int ipart = 0;  ipart < exp_nr_images; ipart++)
		{
			int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
			exp_local_Minvsigma2s[my_image_no].resize(zdim, newdim, newhdim);
			memcpy(exp_local_Minvsigma2s[my_image_no].data, local_Faux_H + my_image_no * image_size, image_size * sizeof(DOUBLE));
	
		}
#ifdef TIMING
		timer.toc(TIMING_DIFF_SHIFT);
#endif
	
		cudaFree(local_Fimgs_D);
		cudaFree(local_Fimg_nomask_D);
		cudaFree(local_myMresol_D);
		cudaFree(sigma2_noise_D);
		cudaFree(group_id_D);
	
		free(exp_local_sqrtXi2_H);
		free(oversampled_translations_H);
		free(group_id_H);
		free(sigma2_noise_H);
		free(local_myMresol);
		free(local_Faux_H);
	
	
	}

void MlOptimiser::doThreadGetSquaredDifferencesAllOrientations_gpu()
{
	// Local variables
	std::vector<DOUBLE> thisthread_min_diff2;
	std::vector< Matrix1D<DOUBLE> > oversampled_orientations, oversampled_translations;
	MultidimArray<Complex > Fref;
	MultidimArray<DOUBLE> Fctf, Minvsigma2;
	Matrix2D<DOUBLE> A;

	// Initialise local mindiff2 for thread-safety
	thisthread_min_diff2.clear();
	thisthread_min_diff2.resize(exp_nr_particles, 99.e99);

	int image_size;

	image_size = shift_img_size;
	DOUBLE* myscale_H , *myscale_D;
	myscale_H = (DOUBLE*) malloc(exp_nr_particles * sizeof(DOUBLE));
	cudaMalloc((void**)&myscale_D, exp_nr_particles * sizeof(DOUBLE));

	long int* my_image_no_list ;
	my_image_no_list = (long*) malloc(sizeof(long) * exp_nr_particles);

	CUFFT_COMPLEX * frefctf_D;

	DOUBLE* exp_highres_Xi2_imgs_H;
	exp_highres_Xi2_imgs_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles);
	DOUBLE* exp_highres_Xi2_imgs_D;
	cudaMalloc((void**)&exp_highres_Xi2_imgs_D, exp_nr_particles * sizeof(DOUBLE));

	long int iorientclass_offset = exp_iclass * exp_nr_rot;

	size_t first_iorient = 0, last_iorient = 0;
	long int nr_orients = sampling.NrDirections() * sampling.NrPsiSamplings();
	last_iorient = nr_orients ;
	float maximal_mem = 0.;
	int orient_stride = nr_orients;
	int nr_orient_loop = 1;

	DOUBLE* A_D, *A_H;
	int nr_A = 0;
	int* valid_orient_trans_index_H, *valid_orient_trans_index_D;
	A_H = (DOUBLE*) malloc(nr_orients * exp_nr_oversampled_rot * 9 * sizeof(DOUBLE));
	cudaMalloc((void**)&A_D, nr_orients * exp_nr_oversampled_rot * 9 * sizeof(DOUBLE));
	valid_orient_trans_index_H = (int*) malloc(exp_nr_particles * nr_orients * exp_nr_trans * sizeof(int));
	cudaMalloc((void**)&valid_orient_trans_index_D, exp_nr_particles * nr_orients * exp_nr_trans * sizeof(int));

	bool* do_proceed_H = (bool*) malloc(sizeof(bool) * exp_nr_particles * exp_nr_trans * nr_orients);
	bool* do_proceed_D;
	cudaMalloc((void**)&do_proceed_D, exp_nr_particles * exp_nr_trans * nr_orients * sizeof(bool));

	CUFFT_COMPLEX * Fref_all_D;
	int f2d_x, f2d_y, data_x, data_y, data_z, data_starty, data_startz;
	/// Now that reference projection has been made loop over someParticles!

	for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			bool is_last_image_in_series = mydata.getNrImagesInSeries(part_id) == (exp_iseries + 1);
			// Which number was this image in the combined array of exp_iseries and part_id
			long int my_image_no = exp_starting_image_no[ipart] + exp_iseries;

			//is_last_image_in_series_H[ipart] = is_last_image_in_series;

			my_image_no_list[ipart] = my_image_no;

			if (do_scale_correction)
			{
				myscale_H[ipart] = mymodel.scale_correction[mydata.getGroupId(part_id, exp_iseries)];
			}

			if ((iter == 1 && do_firstiter_cc) || do_always_cc)
			{
				;
			}
			else
			{
				if (sub_extract)
					exp_highres_Xi2_imgs_H[ipart] = exp_highres_Xi2_imgs_red[my_image_no_list[ipart]];
				else
					exp_highres_Xi2_imgs_H[ipart] = exp_highres_Xi2_imgs[my_image_no_list[ipart]];
			}
		}
	}
	if (do_scale_correction)
	{
		cudaMemcpy(myscale_D, myscale_H, exp_nr_particles * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	}

	if ((iter == 1 && do_firstiter_cc) || do_always_cc)
	{
	}
	else
	{
		cudaMemcpy(exp_highres_Xi2_imgs_D, exp_highres_Xi2_imgs_H, exp_nr_particles * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	}

	f2d_x = shift_img_x;
	f2d_y = shift_img_y;

	data_x = XSIZE(mymodel.PPref[exp_iclass].data);
	data_y = YSIZE(mymodel.PPref[exp_iclass].data);
	data_z = ZSIZE(mymodel.PPref[exp_iclass].data);

	data_starty = STARTINGY(mymodel.PPref[exp_iclass].data);
	data_startz = STARTINGZ(mymodel.PPref[exp_iclass].data);

	DOUBLE* pdf_orientation_H;
	int* is_valid_orientation_H, *is_valid_orientation_D;
	pdf_orientation_H = (DOUBLE*) malloc(sizeof(DOUBLE) * nr_orients);
	is_valid_orientation_H = (int*) malloc(sizeof(int) * nr_orients);
	cudaMalloc((void**)& is_valid_orientation_D, sizeof(int)* nr_orients);
	//The execution path for exp_ipass ==0 and other values
	if (exp_ipass == 0)
	{

		for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
		{
			//long int iorientclass = iorientclass_offset + iorient;
			long int idir = iorient / exp_nr_psi;
			long int ipsi = iorient % exp_nr_psi;
			// Get prior for this direction and skip calculation if prior==0
			//DOUBLE pdf_orientation;
			if (mymodel.orientational_prior_mode == NOPRIOR)
			{
				pdf_orientation_H[iorient] = DIRECT_MULTIDIM_ELEM(mymodel.pdf_direction[exp_iclass], idir);
			}
			else
			{
				pdf_orientation_H[iorient]  = sampling.getPriorProbability(idir, ipsi);
			}
			if (pdf_orientation_H[iorient] > 0.)
			{
				sampling.getOrientations(idir, ipsi, exp_current_oversampling, oversampled_orientations);
				// Loop over all oversampled orientations (only a single one in the first pass)
				for (long int iover_rot = 0; iover_rot < exp_nr_oversampled_rot; iover_rot++)
				{

					// Get the Euler matrix
					Euler_angles2matrix(XX(oversampled_orientations[iover_rot]),
					                    YY(oversampled_orientations[iover_rot]),
					                    ZZ(oversampled_orientations[iover_rot]), A);

					// Take tilt-series into account
					A = (exp_R_mic * A).inv();
					memcpy(A_H + nr_A * 9, A.mdata, 9 * sizeof(DOUBLE));
					nr_A++;
				}

			}
		}
		cudaMemcpy(A_D, A_H, nr_A * 9 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&Fref_all_D, nr_A * image_size * sizeof(CUFFT_COMPLEX ));
		cudaMemset(Fref_all_D, 0., nr_A * image_size * sizeof(CUFFT_COMPLEX ));

		get2DFourierTransform_gpu(Fref_all_D,
		                          A_D,
		                          project_data_D + exp_iclass * (mymodel.PPref[exp_iclass]).data.zyxdim,
		                          IS_INV,
		                          (mymodel.PPref[exp_iclass]).padding_factor,
		                          (mymodel.PPref[exp_iclass]).r_max,
		                          (mymodel.PPref[exp_iclass]).r_min_nn,
		                          f2d_x,
		                          f2d_y,
		                          data_x,
		                          data_y,
		                          data_z,
		                          data_starty,
		                          data_startz,
		                          nr_A,
		                          mymodel.ref_dim);
		cudaDeviceSynchronize();

		maximal_mem = (nr_A * image_size * exp_nr_images * sizeof(CUFFT_COMPLEX )  + nr_A * image_size * sizeof(CUFFT_COMPLEX )) / (1024 * 1024);
		if ((maximal_mem / device_mem_G) > 0.5)
		{

			nr_orient_loop = ROUND(maximal_mem / device_mem_G) + 1;
			orient_stride = (nr_orients + nr_orient_loop - 1) / nr_orient_loop;

		}
		int nr_done_orients = 0;
		int current_valid_orients ;
		for (int nr_loop = 0; nr_loop < nr_orient_loop; nr_loop++)
		{
			first_iorient = nr_loop * orient_stride;
			last_iorient = ((nr_loop + 1) * orient_stride < nr_orients) ? ((nr_loop + 1) * orient_stride) : nr_orients;
			int current_nr_orients = last_iorient - first_iorient;
			current_valid_orients = 0;
			int nr_valid_orient_trans = 0;
			for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
			{

				if (pdf_orientation_H[iorient] > 0.)
				{
					for (int i = 0; i < exp_nr_particles; i++)
					{
						for (int itrans = 0; itrans < exp_nr_trans; itrans++)
						{
							do_proceed_H[i * exp_nr_trans * current_nr_orients + (iorient - first_iorient)* exp_nr_trans + itrans] =  true;
							valid_orient_trans_index_H[nr_valid_orient_trans] = i * exp_nr_trans * current_nr_orients + (iorient - first_iorient) * exp_nr_trans + itrans;
							//valid_orient_trans_index_H[nr_valid_orient_trans] = (iorient-first_iorient)* exp_nr_trans + itrans;
							nr_valid_orient_trans++;
						}
					}
					current_valid_orients++;
					is_valid_orientation_H[iorient - first_iorient] = 1;
				}
				else
				{
					is_valid_orientation_H[iorient - first_iorient] = 0;
				}

			}
			//printf("current_valid_orients = %d\n", current_valid_orients);

			if (current_valid_orients > 0)
			{
				cudaMemcpy(valid_orient_trans_index_D, valid_orient_trans_index_H, nr_valid_orient_trans * sizeof(int), cudaMemcpyHostToDevice);

				cudaMemcpy(do_proceed_D, do_proceed_H, exp_nr_particles * current_nr_orients * exp_nr_trans * sizeof(bool), cudaMemcpyHostToDevice);
				cudaMemcpy(is_valid_orientation_D, is_valid_orientation_H, current_nr_orients * sizeof(int), cudaMemcpyHostToDevice);


				thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast<int> (is_valid_orientation_D);
				thrust::exclusive_scan(dev_ptr,
				                       dev_ptr + current_nr_orients,
				                       dev_ptr);

				cudaMalloc((void**)&frefctf_D, exp_nr_particles * current_valid_orients * exp_nr_oversampled_rot * image_size * sizeof(CUFFT_COMPLEX ));

				calculate_frefctf_gpu(frefctf_D,
				                      Fref_all_D + nr_done_orients * exp_nr_oversampled_rot * image_size,
				                      exp_local_Fctfs_D,
				                      myscale_D,
				                      exp_nr_particles,
				                      current_valid_orients,
				                      exp_nr_oversampled_rot,
				                      image_size,
				                      do_ctf_correction && refs_are_ctf_corrected,
				                      do_scale_correction);
				cudaDeviceSynchronize();
				int diff_offset = (iorientclass_offset + first_iorient) * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans;
				if ((iter == 1 && do_firstiter_cc) || do_always_cc)
				{


					calculate_diff2_no_do_squared_difference_gpu(exp_Mweight_D + diff_offset,
					                                             frefctf_D,
					                                             exp_local_Fimgs_shifted_D,
					                                             exp_local_sqrtXi2_D,
					                                             valid_orient_trans_index_D,
					                                             is_valid_orientation_D,
					                                             exp_nr_particles,
					                                             current_nr_orients,
					                                             exp_nr_trans,
					                                             exp_nr_oversampled_rot,
					                                             exp_nr_oversampled_trans,
					                                             current_valid_orients,
					                                             exp_Mweight_D_size,
					                                             image_size,
					                                             nr_valid_orient_trans);


					cudaDeviceSynchronize();
				}
				else
				{
					calculate_diff2_do_squared_difference_pass0_gpu(exp_Mweight_D + diff_offset,
					                                                frefctf_D,
					                                                exp_local_Fimgs_shifted_D,
					                                                exp_highres_Xi2_imgs_D,
					                                                exp_Minvsigma2s_D,
					                                                valid_orient_trans_index_D,
					                                                exp_nr_particles,
					                                                current_nr_orients,
					                                                exp_nr_trans,
					                                                exp_nr_oversampled_rot,
					                                                exp_nr_oversampled_trans,
					                                                current_valid_orients,
					                                                exp_Mweight_D_size,
					                                                image_size
					                                               );
					cudaDeviceSynchronize();

				}
				cudaFree(frefctf_D);

			}
			nr_done_orients += current_valid_orients;
		}
		free(pdf_orientation_H);
		cudaFree(Fref_all_D);
	}
	else
	{

		for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
		{
			long int iorientclass = iorientclass_offset + iorient;
			long int idir = iorient / exp_nr_psi;
			long int ipsi = iorient % exp_nr_psi;
			// Get prior for this direction and skip calculation if prior==0
			//DOUBLE pdf_orientation;
			if (mymodel.orientational_prior_mode == NOPRIOR)
			{
				pdf_orientation_H[iorient] = DIRECT_MULTIDIM_ELEM(mymodel.pdf_direction[exp_iclass], idir);
			}
			else
			{
				pdf_orientation_H[iorient]  = sampling.getPriorProbability(idir, ipsi);
			}
			if (pdf_orientation_H[iorient] > 0. &&  isSignificantAnyParticleAnyTranslation(iorientclass))
			{
				sampling.getOrientations(idir, ipsi, exp_current_oversampling, oversampled_orientations);
				// Loop over all oversampled orientations (only a single one in the first pass)
				for (long int iover_rot = 0; iover_rot < exp_nr_oversampled_rot; iover_rot++)
				{
					// Get the Euler matrix
					Euler_angles2matrix(XX(oversampled_orientations[iover_rot]),
					                    YY(oversampled_orientations[iover_rot]),
					                    ZZ(oversampled_orientations[iover_rot]), A);

					// Take tilt-series into account
					A = (exp_R_mic * A).inv();
					memcpy(A_H + nr_A * 9, A.mdata, 9 * sizeof(DOUBLE));
					nr_A++;
				}
			}
		}
		cudaMemcpy(A_D, A_H, nr_A * 9 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&Fref_all_D, nr_A * image_size * sizeof(CUFFT_COMPLEX ));
		cudaMemset(Fref_all_D, 0., nr_A * image_size * sizeof(CUFFT_COMPLEX ));
		get2DFourierTransform_gpu(Fref_all_D,
		                          A_D,
		                          project_data_D + exp_iclass * (mymodel.PPref[exp_iclass]).data.zyxdim,
		                          IS_INV,
		                          (mymodel.PPref[exp_iclass]).padding_factor,
		                          (mymodel.PPref[exp_iclass]).r_max,
		                          (mymodel.PPref[exp_iclass]).r_min_nn,
		                          f2d_x,
		                          f2d_y,
		                          data_x,
		                          data_y,
		                          data_z,
		                          data_starty,
		                          data_startz,
		                          nr_A,
		                          mymodel.ref_dim);
		maximal_mem = (nr_A * image_size * exp_nr_images * sizeof(CUFFT_COMPLEX )  + nr_A * image_size * sizeof(CUFFT_COMPLEX )) / (1024 * 1024);
		cudaDeviceSynchronize();

		if ((maximal_mem / device_mem_G) > 0.5)
		{
			nr_orient_loop = ROUND(maximal_mem / device_mem_G) + 1;
			orient_stride = (nr_orients + nr_orient_loop - 1) / nr_orient_loop;
		}
		int nr_done_orients = 0;
		int current_valid_orients ;
		for (int nr_loop = 0; nr_loop < nr_orient_loop; nr_loop++)
		{
			first_iorient = nr_loop * orient_stride;
			last_iorient = ((nr_loop + 1) * orient_stride < nr_orients) ? ((nr_loop + 1) * orient_stride) : nr_orients;
			int current_nr_orients = last_iorient - first_iorient;
			int nr_valid_orient_trans = 0;
			current_valid_orients = 0;

			for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
			{
				long int iorientclass = iorientclass_offset + iorient;
				if (pdf_orientation_H[iorient] > 0. && isSignificantAnyParticleAnyTranslation(iorientclass))
				{
					for (int i = 0; i < exp_nr_particles; i++)
					{
						for (int itrans = 0; itrans < exp_nr_trans; itrans++)
						{
							long int ihidden = (iorientclass_offset + iorient) * exp_nr_trans + itrans;
							do_proceed_H[i * exp_nr_trans * current_nr_orients + (iorient - first_iorient)* exp_nr_trans + itrans] =  DIRECT_A2D_ELEM(exp_Mcoarse_significant, i, ihidden);
							if (DIRECT_A2D_ELEM(exp_Mcoarse_significant, i, ihidden))
							{
								valid_orient_trans_index_H[nr_valid_orient_trans] = i * exp_nr_trans * current_nr_orients + (iorient - first_iorient) * exp_nr_trans + itrans;
								nr_valid_orient_trans++;
								//if(i!=0)
									//std::cout<< "The siginifcant orientation of particles " << i << "  in  "<< ihidden << std::endl;	
							}
						}
					}
					current_valid_orients++;
					is_valid_orientation_H[iorient - first_iorient] = 1;
				}
				else
				{
					is_valid_orientation_H[iorient - first_iorient] = 0;
				}

			}
			if (current_valid_orients > 0)
			{
				cudaMemcpy(valid_orient_trans_index_D, valid_orient_trans_index_H, nr_valid_orient_trans * sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(do_proceed_D, do_proceed_H, exp_nr_particles * current_nr_orients * exp_nr_trans * sizeof(bool), cudaMemcpyHostToDevice);
				cudaMemcpy(is_valid_orientation_D,
				           is_valid_orientation_H,
				           current_nr_orients * sizeof(int),
				           cudaMemcpyHostToDevice);
				thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast<int> (is_valid_orientation_D);
				thrust::exclusive_scan(dev_ptr,
				                       dev_ptr + current_nr_orients,
				                       dev_ptr);
				cudaMemcpy(is_valid_orientation_H,
				           is_valid_orientation_D,
				           current_nr_orients * sizeof(int),
				           cudaMemcpyDeviceToHost);
				cudaMalloc((void**)&frefctf_D, exp_nr_particles * current_valid_orients * exp_nr_oversampled_rot * image_size * sizeof(CUFFT_COMPLEX ));


				calculate_frefctf_gpu(frefctf_D,
				                      Fref_all_D + nr_done_orients * exp_nr_oversampled_rot * image_size,
				                      exp_local_Fctfs_D,
				                      myscale_D,
				                      exp_nr_particles,
				                      current_valid_orients,
				                      exp_nr_oversampled_rot,
				                      image_size,
				                      do_ctf_correction && refs_are_ctf_corrected,
				                      do_scale_correction);

				cudaDeviceSynchronize();
				int diff_offset = (iorientclass_offset + first_iorient) * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans;

				if ((iter == 1 && do_firstiter_cc) || do_always_cc)
				{
					calculate_diff2_no_do_squared_difference_gpu(exp_Mweight_D + diff_offset,
					                                             frefctf_D,
					                                             exp_local_Fimgs_shifted_D,
					                                             exp_local_sqrtXi2_D,
					                                             valid_orient_trans_index_D,
					                                             is_valid_orientation_D,
					                                             exp_nr_particles,
					                                             current_nr_orients,
					                                             exp_nr_trans,
					                                             exp_nr_oversampled_rot,
					                                             exp_nr_oversampled_trans,
					                                             current_valid_orients,
					                                             exp_Mweight_D_size,
					                                             image_size,
					                                             nr_valid_orient_trans);

				}
				else
				{
					calculate_diff2_do_squared_difference_gpu(exp_Mweight_D + diff_offset,
					                                          frefctf_D,
					                                          exp_local_Fimgs_shifted_D,
					                                          exp_highres_Xi2_imgs_D,
					                                          exp_Minvsigma2s_D,
					                                          valid_orient_trans_index_D,
					                                          is_valid_orientation_D,
					                                          exp_nr_particles,
					                                          current_nr_orients,
					                                          exp_nr_trans,
					                                          exp_nr_oversampled_rot,
					                                          exp_nr_oversampled_trans,
					                                          current_valid_orients,
					                                          exp_Mweight_D_size,
					                                          image_size,
					                                          nr_valid_orient_trans);
				}
				cudaFree(frefctf_D);
				cudaDeviceSynchronize();

			}
			nr_done_orients += current_valid_orients;
		}
		free(pdf_orientation_H);
		cudaFree(Fref_all_D);
	}
	free(do_proceed_H);
	free(my_image_no_list);
	free(exp_highres_Xi2_imgs_H);
	free(A_H);
	free(valid_orient_trans_index_H);
	free(myscale_H);
	free(is_valid_orientation_H);

	cudaFree(myscale_D);
	cudaFree(do_proceed_D);
	cudaFree(exp_highres_Xi2_imgs_D);
	cudaFree(A_D);
	cudaFree(is_valid_orientation_D);
	cudaFree(valid_orient_trans_index_D);
}

void MlOptimiser::doThreadGetSquaredDifferencesAllOrientations_multi_class_gpu()
{
	int image_size;

	image_size = shift_img_size;
	DOUBLE* myscale_H , *myscale_D;
	myscale_H = (DOUBLE*) malloc(exp_nr_particles * sizeof(DOUBLE));
	cudaMalloc((void**)&myscale_D, exp_nr_particles * sizeof(DOUBLE));

	long int* my_image_no_list ;
	my_image_no_list = (long*) malloc(sizeof(long) * exp_nr_particles);	

	DOUBLE* exp_highres_Xi2_imgs_H;
	exp_highres_Xi2_imgs_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles);
	DOUBLE* exp_highres_Xi2_imgs_D;
	cudaMalloc((void**)&exp_highres_Xi2_imgs_D, exp_nr_particles * sizeof(DOUBLE));

	long int nr_orients = exp_nr_rot;//sampling.NrDirections() * sampling.NrPsiSamplings();

	DOUBLE*A_D, *A_H;
	
	int* valid_orient_trans_index_H, *valid_orient_trans_index_D;
	A_H = (DOUBLE*) malloc(nr_orients * exp_nr_oversampled_rot * 9 * sizeof(DOUBLE));
	cudaMalloc((void**)&A_D, nr_orients * exp_nr_oversampled_rot * 9 * sizeof(DOUBLE));
	valid_orient_trans_index_H = (int*) malloc(exp_nr_particles * nr_orients * exp_nr_trans * sizeof(int));
	cudaMalloc((void**)&valid_orient_trans_index_D, exp_nr_particles * nr_orients * exp_nr_trans * sizeof(int));

	bool* do_proceed_H = (bool*) malloc(sizeof(bool) * exp_nr_particles * exp_nr_trans * nr_orients);
	bool* do_proceed_D;
	cudaMalloc((void**)&do_proceed_D, exp_nr_particles * exp_nr_trans * nr_orients * sizeof(bool));
	
	DOUBLE* pdf_orientation_H;
	pdf_orientation_H = (DOUBLE*) malloc(sizeof(DOUBLE) * nr_orients);
	int* is_valid_orientation_H, *is_valid_orientation_D;
	is_valid_orientation_H = (int*) malloc(sizeof(int) * nr_orients);
	cudaMalloc((void**)& is_valid_orientation_D, sizeof(int)* nr_orients);
			
	CUFFT_COMPLEX * frefctf_D;
	CUFFT_COMPLEX * Fref_all_D;
			
	for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			bool is_last_image_in_series = mydata.getNrImagesInSeries(part_id) == (exp_iseries + 1);
			// Which number was this image in the combined array of exp_iseries and part_id
			long int my_image_no = exp_starting_image_no[ipart] + exp_iseries;

			//is_last_image_in_series_H[ipart] = is_last_image_in_series;

			my_image_no_list[ipart] = my_image_no;

			if (do_scale_correction)
			{
				myscale_H[ipart] = mymodel.scale_correction[mydata.getGroupId(part_id, exp_iseries)];
			}

			if ((iter == 1 && do_firstiter_cc) || do_always_cc)
			{
				;
			}
			else
			{
				//exp_highres_Xi2_imgs_H[ipart] = exp_highres_Xi2_imgs[my_image_no_list[ipart]];
				if(sub_extract)
						exp_highres_Xi2_imgs_H[ipart] = exp_highres_Xi2_imgs_red[my_image_no_list[ipart]];
				else
						exp_highres_Xi2_imgs_H[ipart] = exp_highres_Xi2_imgs[my_image_no_list[ipart]];
			
			}
		}
	}
	if (do_scale_correction)
	{
		cudaMemcpy(myscale_D, myscale_H, exp_nr_particles * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	}

	if ((iter == 1 && do_firstiter_cc) || do_always_cc)
	{
	}
	else
	{
		cudaMemcpy(exp_highres_Xi2_imgs_D, exp_highres_Xi2_imgs_H, exp_nr_particles * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	}
	std::vector< Matrix1D<DOUBLE> > oversampled_orientations, oversampled_translations;
	//MultidimArray<Complex > Fref;
	//MultidimArray<DOUBLE> Fctf, Minvsigma2;
	Matrix2D<DOUBLE> A;
	
	for (int exp_iclass = iclass_min; exp_iclass <= iclass_max; exp_iclass++)
	{
		if (mymodel.pdf_class[exp_iclass] > 0.)
		{
			// Local variables
			int f2d_x, f2d_y, data_x, data_y, data_z, data_starty, data_startz;
			/// Now that reference projection has been made loop over someParticles!

			size_t first_iorient = 0, last_iorient = 0;
			long int iorientclass_offset = exp_iclass * exp_nr_rot;
			last_iorient = nr_orients ;
			float maximal_mem = 0.;
			int orient_stride = nr_orients;
			int nr_orient_loop = 1;
			int nr_A = 0;
			

			f2d_x = shift_img_x;
			f2d_y = shift_img_y;

			data_x = XSIZE(mymodel.PPref[exp_iclass].data);
			data_y = YSIZE(mymodel.PPref[exp_iclass].data);
			data_z = ZSIZE(mymodel.PPref[exp_iclass].data);

			data_starty = STARTINGY(mymodel.PPref[exp_iclass].data);
			data_startz = STARTINGZ(mymodel.PPref[exp_iclass].data);

			//The execution path for exp_ipass ==0 and other values
			if (exp_ipass == 0)
			{

				for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
				{
					//long int iorientclass = iorientclass_offset + iorient;
					long int idir = iorient / exp_nr_psi;
					long int ipsi = iorient % exp_nr_psi;
					// Get prior for this direction and skip calculation if prior==0
					//DOUBLE pdf_orientation;
					if (mymodel.orientational_prior_mode == NOPRIOR)
					{
						pdf_orientation_H[iorient] = DIRECT_MULTIDIM_ELEM(mymodel.pdf_direction[exp_iclass], idir);
					}
					else
					{
						pdf_orientation_H[iorient]  = sampling.getPriorProbability(idir, ipsi);
					}
					if (pdf_orientation_H[iorient] > 0.)
					{
						sampling.getOrientations(idir, ipsi, exp_current_oversampling, oversampled_orientations);
						// Loop over all oversampled orientations (only a single one in the first pass)
						for (long int iover_rot = 0; iover_rot < exp_nr_oversampled_rot; iover_rot++)
						{

							// Get the Euler matrix
							Euler_angles2matrix(XX(oversampled_orientations[iover_rot]),
							                    YY(oversampled_orientations[iover_rot]),
							                    ZZ(oversampled_orientations[iover_rot]), A);

							// Take tilt-series into account
							A = (exp_R_mic * A).inv();
							memcpy(A_H + nr_A * 9, A.mdata, 9 * sizeof(DOUBLE));
							nr_A++;
						}

					}
				}
				cudaMemcpy(A_D, A_H, nr_A * 9 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
				cudaMalloc((void**)&Fref_all_D, nr_A * image_size * sizeof(CUFFT_COMPLEX ));
				cudaMemset(Fref_all_D, 0., nr_A * image_size * sizeof(CUFFT_COMPLEX ));

				get2DFourierTransform_gpu(Fref_all_D,
				                          A_D,
				                          project_data_D + exp_iclass * (mymodel.PPref[exp_iclass]).data.zyxdim +(sub_extract ? (mymodel.PPref[0]).data.zyxdim*mymodel.nr_classes :0),
				                          IS_INV,
				                          (mymodel.PPref[exp_iclass]).padding_factor,
				                          (mymodel.PPref[exp_iclass]).r_max,
				                          (mymodel.PPref[exp_iclass]).r_min_nn,
				                          f2d_x, 
				                          f2d_y,
				                          data_x,
				                          data_y,
				                          data_z,
				                          data_starty,
				                          data_startz,
				                          nr_A,
				                          mymodel.ref_dim);
				cudaDeviceSynchronize();

				maximal_mem = (nr_A * image_size * exp_nr_images * sizeof(CUFFT_COMPLEX )  + nr_A * image_size * sizeof(CUFFT_COMPLEX )) / (1024 * 1024);
				if ((maximal_mem / device_mem_G) > 0.5)
				{

					nr_orient_loop = ROUND(maximal_mem / device_mem_G) + 1;
					orient_stride = (nr_orients + nr_orient_loop - 1) / nr_orient_loop;

				}
				int nr_done_orients = 0;
				int current_valid_orients ;
				for (int nr_loop = 0; nr_loop < nr_orient_loop; nr_loop++)
				{
					first_iorient = nr_loop * orient_stride;
					last_iorient = ((nr_loop + 1) * orient_stride < nr_orients) ? ((nr_loop + 1) * orient_stride) : nr_orients;
					int current_nr_orients = last_iorient - first_iorient;
					current_valid_orients = 0;
					int nr_valid_orient_trans = 0;
					for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
					{

						if (pdf_orientation_H[iorient] > 0.)
						{
							for (int i = 0; i < exp_nr_particles; i++)
							{
								for (int itrans = 0; itrans < exp_nr_trans; itrans++)
								{
									do_proceed_H[i * exp_nr_trans * current_nr_orients + (iorient - first_iorient)* exp_nr_trans + itrans] =  true;
									valid_orient_trans_index_H[nr_valid_orient_trans] = i * exp_nr_trans * current_nr_orients + (iorient - first_iorient) * exp_nr_trans + itrans;
									//valid_orient_trans_index_H[nr_valid_orient_trans] = (iorient-first_iorient)* exp_nr_trans + itrans;
									nr_valid_orient_trans++;
								}
							}
							current_valid_orients++;
							is_valid_orientation_H[iorient - first_iorient] = 1;
						}
						else
						{
							is_valid_orientation_H[iorient - first_iorient] = 0;
						}

					}
					//printf("current_valid_orient = %d\n", current_valid_orients);

					if (current_valid_orients > 0)
					{
						cudaMemcpy(valid_orient_trans_index_D, valid_orient_trans_index_H, nr_valid_orient_trans * sizeof(int), cudaMemcpyHostToDevice);

						cudaMemcpy(do_proceed_D, do_proceed_H, exp_nr_particles * current_nr_orients * exp_nr_trans * sizeof(bool), cudaMemcpyHostToDevice);
						cudaMemcpy(is_valid_orientation_D, is_valid_orientation_H, current_nr_orients * sizeof(int), cudaMemcpyHostToDevice);


						thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast<int> (is_valid_orientation_D);
						thrust::exclusive_scan(dev_ptr,
						                       dev_ptr + current_nr_orients,
						                       dev_ptr);

						cudaMalloc((void**)&frefctf_D, exp_nr_particles * current_valid_orients * exp_nr_oversampled_rot * image_size * sizeof(CUFFT_COMPLEX ));

						calculate_frefctf_gpu(frefctf_D,
						                      Fref_all_D + nr_done_orients * exp_nr_oversampled_rot * image_size,
						                      exp_local_Fctfs_D,
						                      myscale_D,
						                      exp_nr_particles,
						                      current_valid_orients,
						                      exp_nr_oversampled_rot,
						                      image_size,
						                      do_ctf_correction && refs_are_ctf_corrected,
						                      do_scale_correction);
						cudaDeviceSynchronize();
						int diff_offset = (iorientclass_offset + first_iorient) * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans;
						if ((iter == 1 && do_firstiter_cc) || do_always_cc)
						{


							calculate_diff2_no_do_squared_difference_gpu(exp_Mweight_D + diff_offset,
							                                             frefctf_D,
							                                             exp_local_Fimgs_shifted_D+(sub_extract ?( exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * shift_img_x * shift_img_y ):0),
							                                             exp_local_sqrtXi2_D,
							                                             valid_orient_trans_index_D,
							                                             is_valid_orientation_D,
							                                             exp_nr_particles,
							                                             current_nr_orients,
							                                             exp_nr_trans,
							                                             exp_nr_oversampled_rot,
							                                             exp_nr_oversampled_trans,
							                                             current_valid_orients,
							                                             exp_Mweight_D_size,
							                                             image_size,
							                                             nr_valid_orient_trans);
						


							cudaDeviceSynchronize();
						}
						else
						{
							if(mymodel.ref_dim == 2)
							      calculate_diff2_do_squared_difference_gpu(exp_Mweight_D + diff_offset,
							                                          frefctf_D,
							                                          exp_local_Fimgs_shifted_D+(sub_extract ?( exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * shift_img_x * shift_img_y ):0),
							                                          exp_highres_Xi2_imgs_D,
							                                          exp_Minvsigma2s_D,
							                                          valid_orient_trans_index_D,
							                                          is_valid_orientation_D,
							                                          exp_nr_particles,
							                                          current_nr_orients,
							                                          exp_nr_trans,
							                                          exp_nr_oversampled_rot,
							                                          exp_nr_oversampled_trans,
							                                          current_valid_orients,
							                                          exp_Mweight_D_size,
							                                          image_size,
							                                          nr_valid_orient_trans);
							else
								calculate_diff2_do_squared_difference_pass0_gpu(exp_Mweight_D + diff_offset,
							                                                frefctf_D,
							                                                exp_local_Fimgs_shifted_D+(sub_extract ?( exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * shift_img_x * shift_img_y ):0),
							                                                exp_highres_Xi2_imgs_D,
							                                                exp_Minvsigma2s_D,
							                                                valid_orient_trans_index_D,
							                                                exp_nr_particles,
							                                                current_nr_orients,
							                                                exp_nr_trans,
							                                                exp_nr_oversampled_rot,
							                                                exp_nr_oversampled_trans,
							                                                current_valid_orients,
							                                                exp_Mweight_D_size,
							                                                image_size
							                                               );
							 
							cudaDeviceSynchronize();

						}
						cudaFree(frefctf_D);

					}
					nr_done_orients += current_valid_orients;
				}
				//free(pdf_orientation_H);
				cudaFree(Fref_all_D);
			}
			else
			{

				for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
				{
					long int iorientclass = iorientclass_offset + iorient;
					long int idir = iorient / exp_nr_psi;
					long int ipsi = iorient % exp_nr_psi;
					// Get prior for this direction and skip calculation if prior==0
					//DOUBLE pdf_orientation;
					if (mymodel.orientational_prior_mode == NOPRIOR)
					{
						pdf_orientation_H[iorient] = DIRECT_MULTIDIM_ELEM(mymodel.pdf_direction[exp_iclass], idir);
					}
					else
					{
						pdf_orientation_H[iorient]  = sampling.getPriorProbability(idir, ipsi);
					}
					if (pdf_orientation_H[iorient] > 0. &&  isSignificantAnyParticleAnyTranslation(iorientclass))
					{
						sampling.getOrientations(idir, ipsi, exp_current_oversampling, oversampled_orientations);
						// Loop over all oversampled orientations (only a single one in the first pass)
						for (long int iover_rot = 0; iover_rot < exp_nr_oversampled_rot; iover_rot++)
						{
							// Get the Euler matrix
							Euler_angles2matrix(XX(oversampled_orientations[iover_rot]),
							                    YY(oversampled_orientations[iover_rot]),
							                    ZZ(oversampled_orientations[iover_rot]), A);

							// Take tilt-series into account
							A = (exp_R_mic * A).inv();
							memcpy(A_H + nr_A * 9, A.mdata, 9 * sizeof(DOUBLE));
							nr_A++;
						}
					}
				}
				cudaMemcpy(A_D, A_H, nr_A * 9 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
				cudaMalloc((void**)&Fref_all_D, nr_A * image_size * sizeof(CUFFT_COMPLEX ));
				cudaMemset(Fref_all_D, 0., nr_A * image_size * sizeof(CUFFT_COMPLEX ));
				get2DFourierTransform_gpu(Fref_all_D,
				                          A_D,
								project_data_D + exp_iclass * (mymodel.PPref[exp_iclass]).data.zyxdim +(sub_extract ? (mymodel.PPref[0]).data.zyxdim*mymodel.nr_classes :0),
				                          IS_INV,
				                          (mymodel.PPref[exp_iclass]).padding_factor,
				                          (mymodel.PPref[exp_iclass]).r_max,
				                          (mymodel.PPref[exp_iclass]).r_min_nn,
				                          f2d_x,
				                          f2d_y,
				                          data_x,
				                          data_y,
				                          data_z,
				                          data_starty,
				                          data_startz,
				                          nr_A,
				                          mymodel.ref_dim);
				maximal_mem = (nr_A * image_size * exp_nr_images * sizeof(CUFFT_COMPLEX )  + nr_A * image_size * sizeof(CUFFT_COMPLEX )) / (1024 * 1024);
				cudaDeviceSynchronize();

				if ((maximal_mem / device_mem_G) > 0.5)
				{
					nr_orient_loop = ROUND(maximal_mem / device_mem_G) + 1;
					orient_stride = (nr_orients + nr_orient_loop - 1) / nr_orient_loop;
				}
				int nr_done_orients = 0;
				int current_valid_orients ;
				//printf("current_valid_orients = %d\n", current_valid_orients);
				for (int nr_loop = 0; nr_loop < nr_orient_loop; nr_loop++)
				{
					first_iorient = nr_loop * orient_stride;
					last_iorient = ((nr_loop + 1) * orient_stride < nr_orients) ? ((nr_loop + 1) * orient_stride) : nr_orients;
					int current_nr_orients = last_iorient - first_iorient;
					int nr_valid_orient_trans = 0;
					current_valid_orients = 0;

					for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
					{
						long int iorientclass = iorientclass_offset + iorient;
						if (pdf_orientation_H[iorient] > 0. && isSignificantAnyParticleAnyTranslation(iorientclass))
						{
							for (int i = 0; i < exp_nr_particles; i++)
							{
								for (int itrans = 0; itrans < exp_nr_trans; itrans++)
								{
									long int ihidden = (iorientclass_offset + iorient) * exp_nr_trans + itrans;
									do_proceed_H[i * exp_nr_trans * current_nr_orients + (iorient - first_iorient)* exp_nr_trans + itrans] =  DIRECT_A2D_ELEM(exp_Mcoarse_significant, i, ihidden);
									if (DIRECT_A2D_ELEM(exp_Mcoarse_significant, i, ihidden))
									{
										valid_orient_trans_index_H[nr_valid_orient_trans] = i * exp_nr_trans * current_nr_orients + (iorient - first_iorient) * exp_nr_trans + itrans;
										nr_valid_orient_trans++;
										//if(i!=0)
											//std::cout<< "The siginifcant orientation of particles " << i << "  in  "<< ihidden << std::endl;	
									}
								}
							}
							current_valid_orients++;
							is_valid_orientation_H[iorient - first_iorient] = 1;
						}
						else
						{
							is_valid_orientation_H[iorient - first_iorient] = 0;
						}

					}
					if (current_valid_orients > 0)
					{
						cudaMemcpy(valid_orient_trans_index_D, valid_orient_trans_index_H, nr_valid_orient_trans * sizeof(int), cudaMemcpyHostToDevice);
						cudaMemcpy(do_proceed_D, do_proceed_H, exp_nr_particles * current_nr_orients * exp_nr_trans * sizeof(bool), cudaMemcpyHostToDevice);
						cudaMemcpy(is_valid_orientation_D,
						           is_valid_orientation_H,
						           current_nr_orients * sizeof(int),
						           cudaMemcpyHostToDevice);
						thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast<int> (is_valid_orientation_D);
						thrust::exclusive_scan(dev_ptr,
						                       dev_ptr + current_nr_orients,
						                       dev_ptr);
						cudaMemcpy(is_valid_orientation_H,
						           is_valid_orientation_D,
						           current_nr_orients * sizeof(int),
						           cudaMemcpyDeviceToHost);
						cudaMalloc((void**)&frefctf_D, exp_nr_particles * current_valid_orients * exp_nr_oversampled_rot * image_size * sizeof(CUFFT_COMPLEX ));


						calculate_frefctf_gpu(frefctf_D,
						                      Fref_all_D + nr_done_orients * exp_nr_oversampled_rot * image_size,
						                      exp_local_Fctfs_D,
						                      myscale_D,
						                      exp_nr_particles,
						                      current_valid_orients,
						                      exp_nr_oversampled_rot,
						                      image_size,
						                      do_ctf_correction && refs_are_ctf_corrected,
						                      do_scale_correction);

						cudaDeviceSynchronize();
						int diff_offset = (iorientclass_offset + first_iorient) * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans;

						if ((iter == 1 && do_firstiter_cc) || do_always_cc)
						{
							calculate_diff2_no_do_squared_difference_gpu(exp_Mweight_D + diff_offset,
							                                             frefctf_D,
							                                             exp_local_Fimgs_shifted_D+(sub_extract ?( exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * shift_img_x * shift_img_y ):0),
							                                             exp_local_sqrtXi2_D,
							                                             valid_orient_trans_index_D,
							                                             is_valid_orientation_D,
							                                             exp_nr_particles,
							                                             current_nr_orients,
							                                             exp_nr_trans,
							                                             exp_nr_oversampled_rot,
							                                             exp_nr_oversampled_trans,
							                                             current_valid_orients,
							                                             exp_Mweight_D_size,
							                                             image_size,
							                                             nr_valid_orient_trans);

						}
						else
						{
							calculate_diff2_do_squared_difference_gpu(exp_Mweight_D + diff_offset,
							                                          frefctf_D,
							                                          exp_local_Fimgs_shifted_D+(sub_extract ?( exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * shift_img_x * shift_img_y ):0),
							                                          exp_highres_Xi2_imgs_D,
							                                          exp_Minvsigma2s_D,
							                                          valid_orient_trans_index_D,
							                                          is_valid_orientation_D,
							                                          exp_nr_particles,
							                                          current_nr_orients,
							                                          exp_nr_trans,
							                                          exp_nr_oversampled_rot,
							                                          exp_nr_oversampled_trans,
							                                          current_valid_orients,
							                                          exp_Mweight_D_size,
							                                          image_size,
							                                          nr_valid_orient_trans);
						}
						cudaFree(frefctf_D);
						cudaDeviceSynchronize();

					}
					nr_done_orients += current_valid_orients;
				}
				cudaFree(Fref_all_D);
			}
			
		}
	}
	free(do_proceed_H);
	free(A_H);
	free(valid_orient_trans_index_H);
	free(is_valid_orientation_H);
	free(pdf_orientation_H);

			
	free(myscale_H);
	free(exp_highres_Xi2_imgs_H);			
	free(my_image_no_list);

	cudaFree(myscale_D);
	cudaFree(exp_highres_Xi2_imgs_D);
	cudaFree(do_proceed_D);
	cudaFree(A_D);
	cudaFree(is_valid_orientation_D);
	cudaFree(valid_orient_trans_index_D);
}

	

void MlOptimiser::convertAllSquaredDifferencesToWeights_gpu()
{
	// Convert the squared differences into weights
	// Note there is only one weight for each part_id, because a whole series of images is treated as one particle
	// Initialising...
	exp_sum_weight.resize(exp_nr_particles);
	for (int i = 0; i < exp_nr_particles; i++)
	{
		exp_sum_weight[i] = 0.;
	}

	// Loop from iclass_min to iclass_max to deal with seed generation in first iteration
	exp_iimage = 0;
	exp_ipart = 0;
	long int nr_orients = sampling.NrDirections() * sampling.NrPsiSamplings();
	long int nr_elements = nr_orients * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans;


	if ((iter == 1 && do_firstiter_cc) || do_always_cc)
	{
		//exp_Mweight.resize(exp_nr_particles, mymodel.nr_classes * sampling.NrSamplingPoints(exp_current_oversampling, false));
		bool* exp_Mcoarse_significant_D;

		if (exp_ipass == 0)
		{
			exp_Mcoarse_significant.resize(exp_nr_particles, exp_Mweight_D_size);
			memset(exp_Mcoarse_significant.data, exp_nr_particles * exp_Mweight_D_size * sizeof(bool), 0);
			cudaMalloc((void**)&exp_Mcoarse_significant_D, exp_nr_particles * exp_Mweight_D_size * sizeof(bool));
		}

		calculate_weight_first_iter_gpu(exp_Mweight_D,
		                                exp_Mcoarse_significant_D,
		                                exp_min_diff2_D,
		                                exp_nr_particles,
		                                exp_Mweight_D_size,
		                                iclass_min,
		                                iclass_max,
		                                nr_orients * exp_nr_trans * exp_nr_oversampled_rot * exp_nr_oversampled_trans,
		                                exp_ipass);
		if (exp_ipass == 0)
		{
			cudaMemcpy(exp_Mcoarse_significant.data, exp_Mcoarse_significant_D, exp_nr_particles * exp_Mweight_D_size * sizeof(bool), cudaMemcpyDeviceToHost);
			cudaFree(exp_Mcoarse_significant_D);
		}

		for (long int ori_part_id = exp_my_first_ori_particle, my_image_no = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
		{
			// loop over all particles inside this ori_particle
			for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, exp_ipart++)
			{
				exp_part_id = mydata.ori_particles[ori_part_id].particles_id[i];
				long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];

				exp_thisparticle_sumweight = 0.;


				//DIRECT_A2D_ELEM(exp_Mweight, exp_ipart, myminidx)= 1.;
				exp_thisparticle_sumweight += 1.;
				exp_iimage += mydata.getNrImagesInSeries(part_id);
				exp_sum_weight[exp_ipart] = 1.;

				if (exp_ipass == 0)
				{
					for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++, my_image_no++)
					{
						DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NR_SIGN) = 1.;
					}


				}

				exp_significant_weight[exp_ipart] = 1;
			}
		}
	}
	else
	{
		size_t first_iorient = 0, last_iorient = 0;
		last_iorient = nr_orients - 1;
		int exp_nr_classes = iclass_max - iclass_min + 1;
		int model_nr_classes = mymodel.nr_classes;

		long int xdim_Mweight = exp_Mweight_D_size;

		DOUBLE* pdf_orientation_H, *pdf_orientation_D;
		pdf_orientation_H = (DOUBLE*)malloc(sizeof(DOUBLE) * exp_nr_particles * model_nr_classes * nr_orients);
		cudaMalloc((void**)&pdf_orientation_D, exp_nr_particles * model_nr_classes * nr_orients * sizeof(DOUBLE));

		DOUBLE* pdf_offset_H, *pdf_offset_D;
		pdf_offset_H = (DOUBLE*)malloc(sizeof(DOUBLE) * exp_nr_particles * model_nr_classes * exp_nr_trans);
		cudaMalloc((void**)&pdf_offset_D, exp_nr_particles * model_nr_classes * exp_nr_trans * sizeof(DOUBLE));

		DOUBLE* exp_sum_weight_D, *exp_sum_weight_H;
		cudaMalloc((void**)&exp_sum_weight_D, exp_nr_particles * sizeof(DOUBLE)); //nr_exp_ipart*sizeof(DOUBLE));
		cudaMemset(exp_sum_weight_D, 0., exp_nr_particles * sizeof(DOUBLE));
		exp_sum_weight_H = (DOUBLE*) malloc(sizeof(DOUBLE) * exp_nr_particles);

		for (long int ori_part_id = exp_my_first_ori_particle; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
		{
			// loop over all particles inside this ori_particle
			for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, exp_ipart++)
			{
				if (mydata.ori_particles[ori_part_id].particles_id.size() != 1)
				{
					std::cout << "Warning: mydata.ori_particles[ori_part_id].particles_id.size() != 1" << "	" << mydata.ori_particles[ori_part_id].particles_id.size() << std::endl;
				}

				exp_part_id = mydata.ori_particles[ori_part_id].particles_id[i];

				for (exp_iclass = iclass_min; exp_iclass <= iclass_max; exp_iclass++)
				{

					for (long int iorient = first_iorient; iorient <= last_iorient; iorient++)
					{
						long int idir = iorient / exp_nr_psi;
						long int ipsi = iorient % exp_nr_psi;

						// Get prior for this direction
						if (mymodel.orientational_prior_mode == NOPRIOR)
						{
							pdf_orientation_H[(exp_ipart * model_nr_classes + exp_iclass)*nr_orients + iorient] = DIRECT_MULTIDIM_ELEM(mymodel.pdf_direction[exp_iclass], idir);
						}
						else
						{
							pdf_orientation_H[(exp_ipart * model_nr_classes + exp_iclass)*nr_orients + iorient] = sampling.getPriorProbability(idir, ipsi);
						}
					}
					for (long int itrans = 0; itrans < exp_nr_trans; itrans++)
					{

						// To speed things up, only calculate pdf_offset at the coarse sampling.
						// That should not matter much, and that way one does not need to calculate all the OversampledTranslations
						Matrix1D<DOUBLE> my_offset, my_prior;
						sampling.getTranslation(itrans, my_offset);
						// Convert offsets back to Angstroms to calculate PDF!
						// TODO: if series, then have different exp_old_xoff for each my_image_no....
						// WHAT TO DO WITH THIS?!!!
						if (mymodel.ref_dim == 2)
						{
							pdf_offset_H[(exp_ipart * model_nr_classes + exp_iclass)*exp_nr_trans + itrans] = calculatePdfOffset(exp_old_offset[exp_iimage] + my_offset, mymodel.prior_offset_class[exp_iclass]);
						}
						else
						{
							pdf_offset_H[(exp_ipart * model_nr_classes + exp_iclass)*exp_nr_trans + itrans] = calculatePdfOffset(exp_old_offset[exp_iimage] + my_offset, exp_prior[exp_iimage]);
						}
					}
				}
				// Keep track of number of processed images
				exp_iimage += mydata.getNrImagesInSeries(exp_part_id);
			}

		}
		cudaError cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess)
		{
			printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}

		cudaMemcpy(pdf_orientation_D, pdf_orientation_H, exp_nr_particles * model_nr_classes * nr_orients * sizeof(DOUBLE), cudaMemcpyHostToDevice);
		cudaMemcpy(pdf_offset_D, pdf_offset_H, exp_nr_particles * model_nr_classes * exp_nr_trans * sizeof(DOUBLE), cudaMemcpyHostToDevice);
		DOUBLE* particle_sum_per_weight_D;
		int* none_zero_weight_cound_D;
		DOUBLE* max_weight_D, *max_weight_H, *max_weight_per_particle_D;
		cudaMalloc((void**)&max_weight_D, exp_nr_particles * sizeof(DOUBLE));
		max_weight_H = (DOUBLE*) malloc(exp_nr_particles * sizeof(DOUBLE));
		long int Weight_size_per_class =  (exp_nr_rot*exp_nr_trans*exp_nr_oversampled_rot*exp_nr_oversampled_trans);
		int nr_thread_block = ((iclass_max - iclass_min + 1) * Weight_size_per_class + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128; 
		//sampling.NrSamplingPoints(exp_current_oversampling, false) != (nr_orients*exp_nr_trans*exp_nr_oversampled_rot*exp_nr_oversampled_trans) 
		cudaMalloc((void**)&particle_sum_per_weight_D, nr_thread_block * exp_nr_particles * sizeof(DOUBLE));
		cudaMemset(particle_sum_per_weight_D, 0. , nr_thread_block * exp_nr_particles * sizeof(DOUBLE));
		cudaMalloc((void**)&none_zero_weight_cound_D, nr_thread_block * exp_nr_particles * sizeof(int));
		cudaMemset(none_zero_weight_cound_D, 0. , nr_thread_block * exp_nr_particles * sizeof(int));
		cudaMalloc((void**)&max_weight_per_particle_D, nr_thread_block * exp_nr_particles * sizeof(DOUBLE));

		calculate_weight_gpu(exp_nr_particles,
		                     particle_sum_per_weight_D,
		                     none_zero_weight_cound_D,
		                     max_weight_per_particle_D,
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
		cudaDeviceSynchronize();
		//std::cout << " orient = " << nr_orients << " exp_nr_trans =  " << exp_nr_trans << " exp_nr_oversampled_rot=  " << exp_nr_oversampled_rot \
		//		<< " exp_nr_oversampled_trans= " << exp_nr_oversampled_trans <<" total = " << Weight_size_per_class << std::endl;
		exp_ipart = 0;
		thrust::device_ptr<DOUBLE> exp_Mweight_p(exp_Mweight_D);

		int* sum_num_none_zero_D;
		int sorted_size[exp_nr_particles];
		cudaMalloc((void**)&sum_num_none_zero_D, exp_nr_particles * sizeof(int)); //nr_exp_ipart*sizeof(DOUBLE));
		cudaMemset(sum_num_none_zero_D, 0., exp_nr_particles * sizeof(int));
		calculate_sum_weight_per_particle_gpu(particle_sum_per_weight_D,
		                                      exp_sum_weight_D,
		                                      none_zero_weight_cound_D,
		                                      sum_num_none_zero_D,
		                                      max_weight_per_particle_D,
		                                      max_weight_D,
		                                      nr_thread_block,
		                                      exp_nr_particles,
		                                      nr_thread_block
		                                     );
		cudaMemcpy(exp_sum_weight_H, exp_sum_weight_D, sizeof(DOUBLE) * exp_nr_particles, cudaMemcpyDeviceToHost);
		cudaMemcpy(sorted_size, sum_num_none_zero_D, sizeof(int) * exp_nr_particles, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		//printf("exp_ipass = %d ", exp_ipass);
		for (int i = 0; i < exp_nr_particles; i++)
		{
			exp_sum_weight[i] = exp_sum_weight_H[i];
			//printf("%f ", exp_sum_weight[i]);
			if(exp_sum_weight[i]  == 0)
				std::cout<<"  " <<  exp_sum_weight[i] << "   " << i << " exp_ipass  " << exp_ipass <<std::endl;
		}
		
		//std::cout<< " exp_ipass  " << exp_ipass <<std::endl;

		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess)
		{
			printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}

		if (exp_ipass == 0)
		{
			exp_Mcoarse_significant.resize(exp_nr_particles, exp_Mweight_D_size);
		}

		// Now, for each particle,  find the exp_significant_weight that encompasses adaptive_fraction of exp_sum_weight
		exp_significant_weight.clear();
		exp_significant_weight.resize(exp_nr_particles, 0.);

		int sum_size = 0;
		int max_size = 0;
		for (int i = 0; i < exp_nr_particles; i++)
		{
			sum_size += sorted_size[i];
			/*if (sorted_size[i] > max_size)
			{
				max_size = sorted_size[i];
			}*/
			//std::cout << "sorted_size[i] is " << sorted_size[i]  <<"  " << i << "   " << exp_ipass  << "  " << sum_size << std::endl;
		}
		//DOUBLE* thresh_weight_D;
		//cudaMalloc((void**)&thresh_weight_D, sizeof(DOUBLE) *exp_nr_particles);

		int ss = 0;

		DOUBLE* sorted_weight_D;
		//DOUBLE* sorted_weight_H;
		//sorted_weight_H = (DOUBLE*)malloc(sum_size * sizeof(DOUBLE));
		cudaMalloc((void**)&sorted_weight_D, sum_size * sizeof(DOUBLE));
		cudaMemset( sorted_weight_D, 0., sum_size * sizeof(DOUBLE));
		thrust::device_ptr<DOUBLE> sorted_weight_p(sorted_weight_D);
		
		cudaDeviceSynchronize();

		cudaStat = cudaGetLastError();
		if (cudaStat != cudaSuccess)
		{
			printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
			exit(EXIT_FAILURE);
		}
		bool* exp_Mcoarse_significant_D;
		cudaMalloc((void**)&exp_Mcoarse_significant_D, exp_nr_particles * exp_Mweight_D_size * sizeof(bool));
		thrust::device_ptr<bool> exp_Mcoarse_significant_p(exp_Mcoarse_significant_D);

		for (long int ori_part_id = exp_my_first_ori_particle, my_image_no = 0, exp_ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
		{
			// loop over all particles inside this ori_particle
			for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, exp_ipart++)
			{
				long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
				if(sorted_size[exp_ipart] > 0)
					thrust::copy_if(exp_Mweight_p + xdim_Mweight * exp_ipart + iclass_min * nr_elements, exp_Mweight_p + xdim_Mweight * exp_ipart + (iclass_max + 1)*nr_elements, sorted_weight_p + ss, not_equal_zero());
				cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
					exit(EXIT_FAILURE);
				}
				thrust::sort(sorted_weight_p + ss, sorted_weight_p + ss + sorted_size[exp_ipart], thrust::greater<DOUBLE>());
				thrust::inclusive_scan(sorted_weight_p + ss, sorted_weight_p + sorted_size[exp_ipart] + ss, sorted_weight_p + ss);
				int nr_small = thrust::count_if(sorted_weight_p + ss, sorted_weight_p + ss + sorted_size[exp_ipart], smaller_than(adaptive_fraction * exp_sum_weight[exp_ipart] + (adaptive_fraction * exp_sum_weight[exp_ipart]) * 1e-8));
				cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
					exit(EXIT_FAILURE);
				}
				DOUBLE tmp[2];
				if (nr_small == 0)
				{
					cudaMemcpy(tmp, sorted_weight_D + ss, 1 * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
					exp_significant_weight[exp_ipart] = tmp[0];
					cudaStat = cudaGetLastError();
					if (cudaStat != cudaSuccess)
					{
						std::cout << "paritcle  if "<< exp_ipart  <<"  nr_small  " << nr_small << " significant weight is " << tmp[0] << "  " <<  exp_ipass<<" sorted_size " << sorted_size[exp_ipart]  << std::endl; 
					
						printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s  %d\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat), ss);
						exit(EXIT_FAILURE);
					}
				}
				else if(nr_small == sorted_size[exp_ipart])
				{
					cudaMemcpy(tmp, sorted_weight_D + ss + nr_small - 2, 2 * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
					exp_significant_weight[exp_ipart] = tmp[1] - tmp[0];
					cudaStat = cudaGetLastError();
					if (cudaStat != cudaSuccess)
					{
						printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
						exit(EXIT_FAILURE);
					}
				}
				else
				{
					cudaMemcpy(tmp, sorted_weight_D + ss + nr_small - 1, 2 * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
					exp_significant_weight[exp_ipart] = tmp[1] - tmp[0];
					//std::cout << "paritcle  else "<< exp_ipart  <<"  nr_small  " << nr_small << " significant weight is " << exp_significant_weight[exp_ipart] << "  " <<  exp_ipass<< std::endl; 
					cudaStat = cudaGetLastError();
					if (cudaStat != cudaSuccess)
					{
						printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
						exit(EXIT_FAILURE);
					}
				}
				//std::cout << "paritcle  else "<< exp_ipart  <<"  nr_small  " << nr_small << " sorted_size = " << sorted_size[exp_ipart] << "  " <<  exp_ipass<< std::endl; 
				cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
					exit(EXIT_FAILURE);
				}
				thrust::fill(exp_Mcoarse_significant_p + exp_Mweight_D_size * exp_ipart, exp_Mcoarse_significant_p + exp_Mweight_D_size * (exp_ipart + 1), false);
				cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
					exit(EXIT_FAILURE);
				}
				thrust::replace_if(exp_Mcoarse_significant_p + exp_Mweight_D_size * exp_ipart, exp_Mcoarse_significant_p + exp_Mweight_D_size * (exp_ipart + 1), exp_Mweight_p + exp_Mweight_D_size * exp_ipart, greater_than(exp_significant_weight[exp_ipart] - exp_significant_weight[exp_ipart] * 1e-8), true);
				cudaStat = cudaGetLastError();
				if (cudaStat != cudaSuccess)
				{
					printf("kernel calculate_weight_gpu returned error code %d, line(%d), %s\n", cudaStat, __LINE__, cudaGetErrorString(cudaStat));
					exit(EXIT_FAILURE);
				}
				ss += sorted_size[exp_ipart];
				if (exp_ipass == 0)
				{

					for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++, my_image_no++)
					{
						DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NR_SIGN) = nr_small + 1.;
					}
				}

			}
		}
		//free(sorted_weight_H);
		if (exp_ipass == 0)
		{
			cudaMemcpy(exp_Mcoarse_significant.data, exp_Mcoarse_significant_D, exp_nr_particles * exp_Mweight_D_size * sizeof(bool), cudaMemcpyDeviceToHost);
		}
		cudaFree(pdf_orientation_D);
		cudaFree(pdf_offset_D);
		cudaFree(exp_sum_weight_D);
		cudaFree(particle_sum_per_weight_D);
		cudaFree(none_zero_weight_cound_D);
		cudaFree(sum_num_none_zero_D);
		cudaFree(max_weight_D);
		cudaFree(max_weight_per_particle_D);
		//cudaFree(thresh_weight_D);
		free(pdf_orientation_H);
		free(pdf_offset_H);
		free(exp_sum_weight_H);
		free(max_weight_H);

		cudaFree(sorted_weight_D);
		cudaFree(exp_Mcoarse_significant_D);
	}
}

void MlOptimiser::storeWeightedSums_gpu()
{

#ifdef TIMING
	timer.tic(TIMING_ESP_WSUM);
#endif

	// Initialise the maximum of all weights to a negative value
	exp_max_weight.resize(exp_nr_particles);
	for (int n = 0; n < exp_nr_particles; n++)
	{
		exp_max_weight[n] = -1.;
	}
	//The shifted for no mask images is done in function doThreadPrecalculateShiftedImagesCtfsAndInvSigma2s_gpu
	// In doThreadPrecalculateShiftedImagesCtfsAndInvSigma2s() the origin of the exp_local_Minvsigma2s was omitted.
	// Set those back here

	for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		// loop over all particles inside this ori_particle
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];

			for (exp_iseries = 0; exp_iseries < mydata.getNrImagesInSeries(part_id); exp_iseries++)
			{

				int group_id = mydata.getGroupId(part_id, exp_iseries);
				int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
				DIRECT_MULTIDIM_ELEM(exp_local_Minvsigma2s[my_image_no], 0) = 1. / (sigma2_fudge * DIRECT_A1D_ELEM(mymodel.sigma2_noise[group_id], 0));
			}
		}
	}
	for (exp_iseries = 0; exp_iseries < mydata.getNrImagesInSeries((mydata.ori_particles[exp_my_first_ori_particle]).particles_id[0]); exp_iseries++)
	{
		// TODO: check this!!!
		// I think this is just done for the first ipart
		int my_image_no = exp_starting_image_no[0] + exp_iseries;
		// Get micrograph transformation matrix
		exp_R_mic.resize(3, 3);
		exp_R_mic(0, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_0);
		exp_R_mic(0, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_1);
		exp_R_mic(0, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_2);
		exp_R_mic(1, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_0);
		exp_R_mic(1, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_1);
		exp_R_mic(1, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_2);
		exp_R_mic(2, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_0);
		exp_R_mic(2, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_1);
		exp_R_mic(2, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_2);

		// For norm_correction of this iseries image:
		exp_wsum_norm_correction.resize(exp_nr_particles);
		for (int n = 0; n < exp_nr_particles; n++)
		{
			exp_wsum_norm_correction[n] = 0.;
		}

		// For scale_correction of this iseries image:
		if (do_scale_correction)
		{
			MultidimArray<DOUBLE> aux;
			aux.initZeros(mymodel.ori_size / 2 + 1);
			exp_wsum_scale_correction_XA.resize(exp_nr_particles);
			exp_wsum_scale_correction_AA.resize(exp_nr_particles);
			for (int n = 0; n < exp_nr_particles; n++)
			{
				exp_wsum_scale_correction_XA[n] = aux;
				exp_wsum_scale_correction_AA[n] = aux;
			}
		}

		// Loop from iclass_min to iclass_max to deal with seed generation in first iteration
		/*for (exp_iclass = iclass_min; exp_iclass <= iclass_max; exp_iclass++)
		{
			doThreadStoreWeightedSumsAllOrientations_gpu();

		} // end loop iclass*/
		doThreadStoreWeightedSumsAllOrientations_multi_class_gpu();
		// Extend norm_correction and sigma2_noise estimation to higher resolutions for all particles
		for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
		{
			// loop over all particles inside this ori_particle
			for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
			{
				long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];

				// Which number was this image in the combined array of exp_iseries and exp_part_id
				long int my_image_no = exp_starting_image_no[ipart] + exp_iseries;

				// If the current images were smaller than the original size, fill the rest of wsum_model.sigma2_noise with the power_class spectrum of the images
				int group_id = mydata.getGroupId(part_id, exp_iseries);
				for (int ires = mymodel.current_size / 2 + 1; ires < mymodel.ori_size / 2 + 1; ires++)
				{
					DIRECT_A1D_ELEM(wsum_model.sigma2_noise[group_id], ires) += DIRECT_A1D_ELEM(exp_power_imgs[my_image_no], ires);
					// Also extend the weighted sum of the norm_correction
					exp_wsum_norm_correction[ipart] += DIRECT_A1D_ELEM(exp_power_imgs[my_image_no], ires);
				}

				// Store norm_correction
				// Multiply by old value because the old norm_correction term was already applied to the image
				if (do_norm_correction)
				{
					DOUBLE old_norm_correction = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM);
					old_norm_correction /= mymodel.avg_norm_correction;
					// Now set the new norm_correction in the relevant position of exp_metadata
					// The factor two below is because exp_wsum_norm_correctiom is similar to sigma2_noise, which is the variance for the real/imag components
					// The variance of the total image (on which one normalizes) is twice this value!
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM) = old_norm_correction * sqrt(exp_wsum_norm_correction[ipart] * 2.);
					wsum_model.avg_norm_correction += old_norm_correction * sqrt(exp_wsum_norm_correction[ipart] * 2.);

					if (!(iter == 1 && do_firstiter_cc) && DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM) > 10.)
					{
						std::cout << " WARNING: norm_correction= " << DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM) << " for particle " << part_id << " in group " << group_id + 1 << "; Are your groups large enough?" << std::endl;
						std::cout << " mymodel.current_size= " << mymodel.current_size << " mymodel.ori_size= " << mymodel.ori_size << " part_id= " << part_id << std::endl;
						std::cout << " coarse_size= " << coarse_size << std::endl;
						std::cout << " DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM)= " << DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM) << std::endl;
						std::cout << " mymodel.avg_norm_correction= " << mymodel.avg_norm_correction << std::endl;
						std::cout << " exp_wsum_norm_correction[ipart]= " << exp_wsum_norm_correction[ipart] << std::endl;
						std::cout << " old_norm_correction= " << old_norm_correction << std::endl;
						std::cout << " wsum_model.avg_norm_correction= " << wsum_model.avg_norm_correction << std::endl;
						std::cout << " group_id= " << group_id << " mymodel.scale_correction[group_id]= " << mymodel.scale_correction[group_id] << std::endl;
						std::cout << " mymodel.sigma2_noise[group_id]= " << mymodel.sigma2_noise[group_id] << std::endl;
						std::cout << " wsum_model.sigma2_noise[group_id]= " << wsum_model.sigma2_noise[group_id] << std::endl;
						std::cout << " exp_power_imgs[my_image_no]= " << exp_power_imgs[my_image_no] << std::endl;
						std::cout << " exp_wsum_scale_correction_XA[ipart]= " << exp_wsum_scale_correction_XA[ipart] << " exp_wsum_scale_correction_AA[ipart]= " << exp_wsum_scale_correction_AA[ipart] << std::endl;
						std::cout << " wsum_model.wsum_signal_product_spectra[group_id]= " << wsum_model.wsum_signal_product_spectra[group_id] << " wsum_model.wsum_reference_power_spectra[group_id]= " << wsum_model.wsum_reference_power_spectra[group_id] << std::endl;
						std::cout << " exp_min_diff2[ipart]= " << exp_min_diff2[ipart] << std::endl;
						std::cout << " ml_model.scale_correction[group_id]= " << mymodel.scale_correction[group_id] << std::endl;
						std::cout << " exp_significant_weight[ipart]= " << exp_significant_weight[ipart] << std::endl;
						std::cout << " exp_max_weight[ipart]= " << exp_max_weight[ipart] << std::endl;

					}
				}

				// Store weighted sums for scale_correction
				if (do_scale_correction)
				{
					// Divide XA by the old scale_correction and AA by the square of that, because was incorporated into Fctf
					exp_wsum_scale_correction_XA[ipart] /= mymodel.scale_correction[group_id];
					exp_wsum_scale_correction_AA[ipart] /= mymodel.scale_correction[group_id] * mymodel.scale_correction[group_id];

					wsum_model.wsum_signal_product_spectra[group_id] += exp_wsum_scale_correction_XA[ipart];
					wsum_model.wsum_reference_power_spectra[group_id] += exp_wsum_scale_correction_AA[ipart];
				}

			} // end loop part_id (i)
		} // end loop ori_part_id

	} // end loop exp_iseries

#ifdef DEBUG_OVERSAMPLING
	std::cerr << " max_weight= " << max_weight << " nr_sign_sam= " << nr_significant_samples << " sign w= " << exp_significant_weight << std::endl;
#endif

	// Some analytics...
	// Calculate normalization constant for dLL
	for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		// loop over all particles inside this ori_particle
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			DOUBLE logsigma2 = 0.;
			for (long int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
			{
				int group_id = mydata.getGroupId(part_id, iseries);
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Mresol_fine)
				{
					int ires = DIRECT_MULTIDIM_ELEM(Mresol_fine, n);
					// Note there is no sqrt in the normalisation term because of the 2-dimensionality of the complex-plane
					// Also exclude origin from logsigma2, as this will not be considered in the P-calculations
					if (ires > 0)
					{
						logsigma2 += log(2. * PI * DIRECT_A1D_ELEM(mymodel.sigma2_noise[group_id], ires));
					}
				}

			}

			if (exp_sum_weight[ipart] == 0)
			{
				std::cerr << " part_id= " << part_id << std::endl;
				std::cerr << " ipart= " << ipart << std::endl;
				std::cerr << " exp_min_diff2[ipart]= " << exp_min_diff2[ipart] << std::endl;
				std::cerr << " logsigma2= " << logsigma2 << std::endl;
				int group_id = mydata.getGroupId(part_id, 0);
				std::cerr << " group_id= " << group_id << std::endl;
				std::cerr << " ml_model.scale_correction[group_id]= " << mymodel.scale_correction[group_id] << std::endl;
				std::cerr << " exp_significant_weight[ipart]= " << exp_significant_weight[ipart] << std::endl;
				std::cerr << " exp_max_weight[ipart]= " << exp_max_weight[ipart] << std::endl;
				std::cerr << " ml_model.sigma2_noise[group_id]= " << mymodel.sigma2_noise[group_id] << std::endl;
				REPORT_ERROR("ERROR: exp_sum_weight[ipart]==0");
			}

			DOUBLE dLL;

			if ((iter == 1 && do_firstiter_cc) || do_always_cc)
			{
				dLL = -exp_min_diff2[ipart];
			}
			else
			{
				dLL = log(exp_sum_weight[ipart]) - exp_min_diff2[ipart] - logsigma2;
			}

			wsum_model.LL += dLL;
			wsum_model.ave_Pmax += DIRECT_A2D_ELEM(exp_metadata, exp_starting_image_no[ipart], METADATA_PMAX);

			// Also store dLL of each image in the output array
			for (long int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
			{
				long int my_image_no = exp_starting_image_no[ipart] + iseries;
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_DLL) = dLL;
			}
		} // end loop part_id
	} // end loop ori_part_id
#ifdef TIMING
	timer.toc(TIMING_ESP_WSUM);
#endif

}

void MlOptimiser::doThreadStoreWeightedSumsAllOrientations_gpu()
{

	std::vector< Matrix1D<DOUBLE> > oversampled_orientations, oversampled_translations;
	Matrix2D<DOUBLE> A;

	MultidimArray<DOUBLE> Minvsigma2;
	DOUBLE rot, tilt, psi;
	bool have_warned_small_scale = false;
	std::vector< MultidimArray<DOUBLE> > Mctf_Array;

	Mctf_Array.clear();
	Mctf_Array.resize(exp_nr_particles);
	for (int i = 0; i < exp_nr_particles; i++)
	{
		Mctf_Array[i].resize(exp_Fimgs[0]);
		Mctf_Array[i].initConstant(1.);
	}

	// Initialise Minvsigma2 to all-1 for if !do_map
	Minvsigma2.resize(exp_Fimgs[0]);
	Minvsigma2.initConstant(1.);

	// Make local copies of weighted sums (excepts BPrefs, which are too big)
	// so that there are not too many mutex locks below
	std::vector<MultidimArray<DOUBLE> > thr_wsum_sigma2_noise, thr_wsum_scale_correction_XA, thr_wsum_scale_correction_AA, thr_wsum_pdf_direction;
	std::vector<DOUBLE> thr_wsum_norm_correction, thr_sumw_group, thr_wsum_pdf_class, thr_wsum_prior_offsetx_class, thr_wsum_prior_offsety_class, thr_max_weight;
	DOUBLE thr_wsum_sigma2_offset;
	MultidimArray<DOUBLE> thr_metadata, zeroArray;

	//==========================================================================
	//GPU computing related data structures
	CUFFT_COMPLEX *  frefctf_D, *Fimg_D;
	DOUBLE* mctf_D, *mctf_H, *mctf_out_D;
	DOUBLE* myscale_D, *myscale_H;
	DOUBLE* fweight_D;
	DOUBLE* fimg_H, *fweight_H;
	DOUBLE* exp_significant_sum_weight_D, *exp_significant_sum_weight_H;
	//DOUBLE* oversampled_translations_H;
	DOUBLE* exp_local_weight_D;
	int*  group_id_H,  *group_id_D;
	DOUBLE* thr_wsum_sigma2_noise_D, *thr_wsum_norm_correction_D,
	        *thr_wsum_scale_correction_XA_D, *thr_wsum_scale_correction_AA_D;

	DOUBLE* thr_wsum_sigma2_noise_H, *thr_wsum_norm_correction_H,
	        *thr_wsum_scale_correction_XA_H, *thr_wsum_scale_correction_AA_H;
	DOUBLE* data_vs_prior_class_D;
	int* mresol_fine_D;
	DOUBLE* exp_old_offset_D, *exp_old_offset_H;
	DOUBLE* exp_prior_H, *exp_prior_D;
	DOUBLE* exp_Minvsigma2s_H;
	DOUBLE* A_array_D, *A_array_H;
	int* isSignificant_H, *isSignificant_D, *Significant_list_H, *Significant_list_D;
	int  f2d_x, f2d_y, data_x, data_y, data_z, data_starty, data_startz;

	f2d_x = shift_img_x;
	f2d_y = shift_img_y;

	data_x = XSIZE(mymodel.PPref[exp_iclass].data);
	data_y = YSIZE(mymodel.PPref[exp_iclass].data);
	data_z = ZSIZE(mymodel.PPref[exp_iclass].data);

	data_starty = STARTINGY(mymodel.PPref[exp_iclass].data);
	data_startz = STARTINGZ(mymodel.PPref[exp_iclass].data);
	int nr_A = 0 ;
	int image_size = shift_img_size;//exp_local_Fimgs_shifted[0].zyxdim;
	long int nr_orients = exp_nr_rot;//sampling.NrDirections() * sampling.NrPsiSamplings();

	exp_old_offset_H = (DOUBLE*)malloc(sizeof(DOUBLE) * 2 * exp_nr_images);
	exp_prior_H = (DOUBLE*)malloc(sizeof(DOUBLE) * 2 * exp_nr_images);
	fimg_H = (DOUBLE*) malloc(exp_Fimgs[0].zyxdim * sizeof(CUFFT_COMPLEX ));
	fweight_H = (DOUBLE*) malloc(exp_Fimgs[0].zyxdim * sizeof(DOUBLE));
	mctf_H = (DOUBLE*) malloc(exp_Fimgs[0].zyxdim * exp_nr_images * sizeof(DOUBLE));
	myscale_H = (DOUBLE*) malloc(exp_nr_images * sizeof(DOUBLE));
	exp_significant_sum_weight_H = (DOUBLE*)malloc(exp_nr_images * 2 * sizeof(DOUBLE));
	//oversampled_translations_H = (DOUBLE*) malloc(exp_nr_oversampled_rot * exp_nr_images * exp_nr_trans * exp_nr_oversampled_trans * exp_Fimgs[0].getDim() * sizeof(DOUBLE));
	group_id_H = (int*)malloc(exp_nr_images * sizeof(int));
	thr_wsum_sigma2_noise_H = (DOUBLE*)malloc(mymodel.nr_groups * (mymodel.ori_size / 2 + 1) * sizeof(DOUBLE));
	thr_wsum_norm_correction_H = (DOUBLE*)malloc(exp_nr_particles * sizeof(DOUBLE));
	thr_wsum_scale_correction_XA_H = (DOUBLE*)malloc(exp_nr_particles * (mymodel.ori_size / 2 + 1) * sizeof(DOUBLE));
	thr_wsum_scale_correction_AA_H = (DOUBLE*)malloc(exp_nr_particles * (mymodel.ori_size / 2 + 1) * sizeof(DOUBLE));
	exp_Minvsigma2s_H = (DOUBLE*) malloc(exp_nr_images * exp_local_Minvsigma2s[0].zyxdim * sizeof(DOUBLE));
	A_array_H = (DOUBLE*)malloc(sizeof(DOUBLE) * nr_orients * exp_nr_oversampled_rot * 9);
	isSignificant_H = (int*) malloc(nr_orients * sizeof(int));
	Significant_list_H = (int*) malloc(nr_orients * sizeof(int));
	cudaMalloc((void**)&mctf_D, exp_Fimgs[0].zyxdim * exp_nr_images * sizeof(DOUBLE));
	cudaMalloc((void**)&myscale_D, exp_nr_images * sizeof(DOUBLE));
	cudaMalloc((void**)&exp_significant_sum_weight_D, exp_nr_images * 2 * sizeof(DOUBLE));
	cudaMalloc((void**)&group_id_D, exp_nr_images * sizeof(int));

	cudaMalloc((void**)&mresol_fine_D, Mresol_fine.zyxdim * sizeof(int));
	cudaMalloc((void**)&data_vs_prior_class_D, mymodel.data_vs_prior_class[exp_iclass].zyxdim * sizeof(DOUBLE));
	cudaMalloc((void**)&thr_wsum_sigma2_noise_D, mymodel.nr_groups * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMalloc((void**)&thr_wsum_norm_correction_D, exp_nr_particles * sizeof(DOUBLE));
	cudaMalloc((void**)&thr_wsum_scale_correction_XA_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMalloc((void**)&thr_wsum_scale_correction_AA_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));

	cudaMemset(thr_wsum_sigma2_noise_D, 0. , mymodel.nr_groups * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMemset(thr_wsum_norm_correction_D, 0. , exp_nr_particles * sizeof(DOUBLE));
	cudaMemset(thr_wsum_scale_correction_XA_D, 0. , exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMemset(thr_wsum_scale_correction_AA_D, 0. , exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));

	cudaMalloc((void**)&exp_old_offset_D, sizeof(DOUBLE) * 2 * exp_nr_images);
	cudaMalloc((void**)&exp_prior_D, sizeof(DOUBLE) * 2 * exp_nr_images);
	cudaMalloc((void**)&isSignificant_D, nr_orients * sizeof(int));
	cudaMalloc((void**)&A_array_D, sizeof(DOUBLE) * nr_orients * exp_nr_oversampled_rot * 9);
	cudaMalloc((void**)&Significant_list_D, nr_orients * sizeof(int));
	//============================================================================
	// Wsum_sigma_noise2 is a 1D-spectrum for each group
	zeroArray.initZeros(mymodel.ori_size / 2 + 1);
	thr_wsum_sigma2_noise.resize(mymodel.nr_groups);
	for (int n = 0; n < mymodel.nr_groups; n++)
	{
		thr_wsum_sigma2_noise[n] = zeroArray;
	}
	// scale-correction terms are a spectrum for each particle
	thr_wsum_scale_correction_XA.resize(exp_nr_particles);
	thr_wsum_scale_correction_AA.resize(exp_nr_particles);
	for (int n = 0; n < exp_nr_particles; n++)
	{
		thr_wsum_scale_correction_XA[n] = zeroArray;
		thr_wsum_scale_correction_AA[n] = zeroArray;
	}
	// wsum_pdf_direction is a 1D-array (of length sampling.NrDirections(0, true)) for each class
	zeroArray.initZeros(sampling.NrDirections(0, true));
	thr_wsum_pdf_direction.resize(mymodel.nr_classes);
	for (int n = 0; n < mymodel.nr_classes; n++)
	{
		thr_wsum_pdf_direction[n] = zeroArray;
	}
	// wsum_norm_correction is a DOUBLE for each particle
	thr_wsum_norm_correction.resize(exp_nr_particles, 0.);
	// sumw_group is a DOUBLE for each group
	thr_sumw_group.resize(mymodel.nr_groups, 0.);
	// wsum_pdf_class is a DOUBLE for each class
	thr_wsum_pdf_class.resize(mymodel.nr_classes, 0.);
	if (mymodel.ref_dim == 2)
	{
		thr_wsum_prior_offsetx_class.resize(mymodel.nr_classes, 0.);
		thr_wsum_prior_offsety_class.resize(mymodel.nr_classes, 0.);
	}
	// max_weight is a DOUBLE for each particle
	thr_max_weight.resize(exp_nr_particles, 0.);
	// wsum_sigma2_offset is just a DOUBLE
	thr_wsum_sigma2_offset = 0.;
	// metadata is a 2D array of nr_particles x METADATA_LINE_LENGTH
	thr_metadata.initZeros(exp_metadata);

	// exp_iclass loop does not always go from 0 to nr_classes!
	long int iorientclass_offset = exp_iclass * exp_nr_rot;
	size_t first_iorient = 0, last_iorient = nr_orients;

	for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		// loop over all particles inside this ori_particle
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			// Which number was this image in the combined array of iseries and part_idpart_id
			long int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
			int group_id = mydata.getGroupId(part_id, exp_iseries);
			group_id_H[my_image_no] = group_id;

			exp_old_offset_H[my_image_no] = XX(exp_old_offset[my_image_no]);
			exp_old_offset_H[my_image_no + exp_nr_images] = YY(exp_old_offset[my_image_no]);

			exp_prior_H[my_image_no] = XX(exp_prior[my_image_no]);
			exp_prior_H[my_image_no + exp_nr_images] = YY(exp_prior[my_image_no]);

			if (!do_skip_maximization)
			{
				// Apply CTF to reference projection
				if (do_ctf_correction)
				{
					Mctf_Array[my_image_no]  = exp_local_Fctfs[my_image_no];
				}
				else
				{
					// initialise because there are multiple particles and Mctf gets selfMultiplied for scale_correction
					Mctf_Array[my_image_no] .initConstant(1.);
				}

				memcpy(mctf_H + my_image_no * exp_Fimgs[0].zyxdim, Mctf_Array[my_image_no].data, exp_Fimgs[0].zyxdim * sizeof(DOUBLE));

				if (do_scale_correction)
				{
					// TODO: implemenent B-factor as well...
					DOUBLE myscale = mymodel.scale_correction[group_id];
					if (myscale > 10000.)
					{
						std::cerr << " rlnMicrographScaleCorrection= " << myscale << " group= " << group_id + 1 << " my_image_no= " << my_image_no << std::endl;
						REPORT_ERROR("ERROR: rlnMicrographScaleCorrection is very high. Did you normalize your data?");
					}
					else if (myscale < 0.001)
					{

						if (!have_warned_small_scale)
						{
							std::cout << " WARNING: ignoring group " << group_id + 1 << " with very small or negative scale (" << myscale <<
							          "); Use larger groups for more stable scale estimates." << std::endl;
							have_warned_small_scale = true;
						}
						myscale = 0.001;
					}
					myscale_H[my_image_no] = myscale;
				}
			} // end if !do_skip_maximization
			exp_significant_sum_weight_H[ipart] = exp_significant_weight[ipart];
			exp_significant_sum_weight_H[ipart + exp_nr_images] = exp_sum_weight[ipart];
		}
	}
	cudaMemcpy(exp_old_offset_D, exp_old_offset_H, sizeof(DOUBLE) * 2 * exp_nr_images, cudaMemcpyHostToDevice);
	cudaMemcpy(exp_prior_D, exp_prior_H, sizeof(DOUBLE) * 2 * exp_nr_images, cudaMemcpyHostToDevice);
	cudaMemcpy(mctf_D, mctf_H, exp_nr_images * exp_Fimgs[0].zyxdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cudaMemcpy(myscale_D, myscale_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cudaMemcpy(group_id_D, group_id_H, exp_nr_images * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(exp_significant_sum_weight_D, exp_significant_sum_weight_H, exp_nr_images * 2 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	//================================================================
	for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		// loop over all particles inside this ori_particle
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			// Which number was this image in the combined array of iseries and part_idpart_id
			long int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
			if (do_map)
			{
				memcpy(exp_Minvsigma2s_H + my_image_no * exp_local_Minvsigma2s[0].zyxdim, exp_local_Minvsigma2s[my_image_no].data, exp_local_Minvsigma2s[0].zyxdim * sizeof(DOUBLE));
			}
			else
			{
				memcpy(exp_Minvsigma2s_H + my_image_no * exp_local_Minvsigma2s[0].zyxdim, Minvsigma2.data, exp_local_Minvsigma2s[0].zyxdim * sizeof(DOUBLE));
			}
		}
	}
	cudaMemcpy(exp_Minvsigma2s_D, exp_Minvsigma2s_H, exp_nr_images * exp_Fimgs[0].yxdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cudaMemcpy(mresol_fine_D, Mresol_fine.data, Mresol_fine.zyxdim * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(data_vs_prior_class_D, mymodel.data_vs_prior_class[exp_iclass].data, mymodel.data_vs_prior_class[exp_iclass].zyxdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	//================================================================
	for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
	{

		long int iorientclass = iorientclass_offset + iorient;
		isSignificant_H[iorient] = isSignificantAnyParticleAnyTranslation(iorientclass) ? 1 : 0;
		// Only proceed if any of the particles had any significant coarsely sampled translation
		if (isSignificantAnyParticleAnyTranslation(iorientclass))
		{

			long int idir = iorient / exp_nr_psi;
			long int ipsi = iorient % exp_nr_psi;

			// Now get the oversampled (rot, tilt, psi) triplets
			// This will be only the original (rot,tilt,psi) triplet if (adaptive_oversampling==0)
			sampling.getOrientations(idir, ipsi, adaptive_oversampling, oversampled_orientations);

			// Loop over all oversampled orientations (only a single one in the first pass)
			for (long int iover_rot = 0; iover_rot < exp_nr_oversampled_rot; iover_rot++)
			{
				rot = XX(oversampled_orientations[iover_rot]);
				tilt = YY(oversampled_orientations[iover_rot]);
				psi = ZZ(oversampled_orientations[iover_rot]);
				// Get the Euler matrix
				Euler_angles2matrix(rot, tilt, psi, A);

				// Take tilt-series into account
				A = (exp_R_mic * A).inv();
				memcpy(A_array_H + (nr_A + iover_rot) * 9, A.mdata, sizeof(DOUBLE) * 9);
			}
			nr_A += exp_nr_oversampled_rot;
		}
	}
	if (nr_A > 0)
	{

		CUFFT_COMPLEX * Fref_global_D;

		DOUBLE* oversampled_translations_H, *oversampled_translations_D;
		DOUBLE* pointer_dir_nonzeroprior_H, *pointer_dir_nonzeroprior_D;
		DOUBLE* thr_wsum_pdf_direction_D, *thr_wsum_pdf_direction_H;

		DOUBLE* thr_sumw_group_D, *thr_sumw_group_H;
		DOUBLE* thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D;
		int nr_done_orients = 0;
		int current_nr_valid_orients ;
		DOUBLE*  thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H;

		thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H = (DOUBLE*) malloc(4 * sizeof(DOUBLE));
		thr_wsum_pdf_direction_H = (DOUBLE*)malloc(thr_wsum_pdf_direction[exp_iclass].xdim * sizeof(DOUBLE));
		pointer_dir_nonzeroprior_H = (DOUBLE*)malloc(sampling.pointer_dir_nonzeroprior.size() * sizeof(DOUBLE));
		oversampled_translations_H = (DOUBLE*)malloc(exp_nr_trans * exp_nr_oversampled_trans * 2 * sizeof(DOUBLE));
		thr_sumw_group_H = (DOUBLE*) malloc(sizeof(DOUBLE) * mymodel.nr_groups);

		cudaMalloc((void**)&pointer_dir_nonzeroprior_D, sampling.pointer_dir_nonzeroprior.size() * sizeof(DOUBLE));
		cudaMalloc((void**)&thr_wsum_pdf_direction_D, thr_wsum_pdf_direction[exp_iclass].xdim * sizeof(DOUBLE));
		cudaMemset(thr_wsum_pdf_direction_D, 0., thr_wsum_pdf_direction[exp_iclass].xdim * sizeof(DOUBLE));
		cudaMalloc((void**)&oversampled_translations_D, exp_nr_trans * exp_nr_oversampled_trans * 2 * sizeof(DOUBLE));
		cudaMalloc((void**)&thr_sumw_group_D, sizeof(DOUBLE) * mymodel.nr_groups);
		cudaMemset(thr_sumw_group_D, 0., mymodel.nr_groups * sizeof(DOUBLE));
		cudaMalloc((void**)&thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D, 4 * sizeof(DOUBLE));
		cudaMemset(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D, 0., 4 * sizeof(DOUBLE));

		for (long int itrans = 0; itrans < exp_nr_trans; itrans++)
		{
			sampling.getTranslations(itrans, adaptive_oversampling, oversampled_translations);
			for (int i = 0; i < exp_nr_oversampled_trans; i++)
			{
				int offset = itrans * exp_nr_oversampled_trans;
				oversampled_translations_H[ i  + offset] = XX(oversampled_translations[i]);
				oversampled_translations_H[ i  + offset + exp_nr_trans * exp_nr_oversampled_trans] = YY(oversampled_translations[i]);
			}
		}
		cudaMemcpy(oversampled_translations_D, oversampled_translations_H, exp_nr_trans * exp_nr_oversampled_trans * 2 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
		if (mymodel.orientational_prior_mode != NOPRIOR)
		{
			for (int i = 0; i < sampling.pointer_dir_nonzeroprior.size(); i++)
			{
				pointer_dir_nonzeroprior_H[i] = sampling.getDirectionNumberAlsoZeroPrior(i);
			}
			cudaMemcpy(pointer_dir_nonzeroprior_D, pointer_dir_nonzeroprior_H, sampling.pointer_dir_nonzeroprior.size()*sizeof(DOUBLE), cudaMemcpyHostToDevice);

		}

		cudaMemcpy(A_array_D, A_array_H, sizeof(DOUBLE) * nr_A * 9, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&Fref_global_D, nr_A * image_size * sizeof(CUFFT_COMPLEX ));

		cudaMemset(Fref_global_D, 0., nr_A * image_size * sizeof(CUFFT_COMPLEX ));
		get2DFourierTransform_gpu(Fref_global_D,
		                          A_array_D,
		                          project_data_D + exp_iclass * (mymodel.PPref[exp_iclass]).data.zyxdim,
		                          IS_INV,
		                          (mymodel.PPref[exp_iclass]).padding_factor,
		                          (mymodel.PPref[exp_iclass]).r_max,
		                          (mymodel.PPref[exp_iclass]).r_min_nn,
		                          f2d_x,
		                          f2d_y,
		                          data_x,
		                          data_y,
		                          data_z,
		                          data_starty,
		                          data_startz,
		                          nr_A,
		                          mymodel.ref_dim);
		cudaDeviceSynchronize();

		//Calculate the number of loops for all orientations in the current class
		//occording to the memmory requirement and the total memory space of GPU
		int orient_stride = nr_orients;
		int nr_orient_loop = 1;
		//Calculate the maximal memory requirement for the current orient
		float weight_mem_size = (exp_nr_images * sizeof(DOUBLE) * exp_Mweight_D_size / (1024 * 1024));
		float usage_mem = weight_mem_size + nr_A * image_size * sizeof(CUFFT_COMPLEX ) / (1024 * 1024);
		//Calculate the maximal memory requirement for the current orient
		float required_mem = (nr_A * image_size * exp_nr_images * sizeof(DOUBLE) * 3  + nr_A * image_size * sizeof(DOUBLE) * 3 + weight_mem_size) / (1024 * 1024);
		//If the require memory space is largere than 75% of the total GPU device memory
		//We loop all the orients of the corrent class through several steps
		if ((required_mem / (device_mem_G - usage_mem)) > 0.6)
		{

			nr_orient_loop = ROUND(required_mem / (device_mem_G - usage_mem)) + 1;
			orient_stride = (nr_orients + nr_orient_loop - 1) / nr_orient_loop;
			std::cout << "warning: the total global memory  is about " << device_mem_G << " MB and usaged is : " <<  usage_mem << std::endl;
			std::cout << "warning: the maximal memory requirement in the current  orientationsi about" << required_mem << " MB" << "  stride is " << orient_stride <<   std::endl;
		}
		for (int nr_loop = 0; nr_loop < nr_orient_loop; nr_loop++)
		{
			first_iorient = nr_loop * orient_stride;
			last_iorient = ((nr_loop + 1) * orient_stride < nr_orients) ? ((nr_loop + 1) * orient_stride) : nr_orients;
			int current_nr_orients = last_iorient - first_iorient;
			current_nr_valid_orients = 0;
			//int nr_valid_orient_trans = 0;

			for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
			{
				// Only proceed if any of the particles had any significant coarsely sampled translation
				if (isSignificant_H[iorient])
				{
					Significant_list_H[current_nr_valid_orients] = iorient;
					current_nr_valid_orients++;
				}
			}
			//There are valid orients in the current loop
			if (current_nr_valid_orients > 0)
			{

				cudaMalloc((void**)&Fimg_D, current_nr_valid_orients * exp_nr_oversampled_rot * image_size * sizeof(CUFFT_COMPLEX ));
				cudaMalloc((void**)&fweight_D, current_nr_valid_orients * exp_nr_oversampled_rot * image_size * sizeof(DOUBLE));

				cudaMalloc((void**)&frefctf_D, current_nr_valid_orients * exp_nr_oversampled_rot * exp_Fimgs[0].zyxdim * exp_nr_images * sizeof(CUFFT_COMPLEX ));
				cudaMalloc((void**)&mctf_out_D,    current_nr_valid_orients * exp_nr_oversampled_rot * exp_Fimgs[0].zyxdim * exp_nr_images * sizeof(DOUBLE));

				cudaMemset(Fimg_D, 0. , current_nr_valid_orients * exp_nr_oversampled_rot * exp_Fimgs[0].zyxdim * sizeof(CUFFT_COMPLEX ));
				cudaMemset(fweight_D, 0. , current_nr_valid_orients * exp_nr_oversampled_rot * exp_Fimgs[0].zyxdim * sizeof(DOUBLE));

				if (!do_skip_maximization)
				{
					calculate_frefctf_Mctf_gpu(frefctf_D,
					                           Fref_global_D + nr_done_orients * exp_nr_oversampled_rot * image_size,
					                           mctf_D,
					                           mctf_out_D,
					                           myscale_D,
					                           current_nr_valid_orients,
					                           exp_nr_oversampled_rot,
					                           exp_nr_images,
					                           exp_Fimgs[0].zyxdim,
					                           do_ctf_correction && refs_are_ctf_corrected,
					                           do_scale_correction);
				}

				cudaDeviceSynchronize();
				int* valid_weight_list_D, *compact_position_list_D;
				cudaMalloc((void**)&valid_weight_list_D, exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans * sizeof(int));
				cudaMalloc((void**)&compact_position_list_D, exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans * sizeof(int));

				cudaMemset(valid_weight_list_D, 0, exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans * sizeof(int));
				cudaMemcpy(isSignificant_D, isSignificant_H + first_iorient, current_nr_orients * sizeof(int), cudaMemcpyHostToDevice);
				sign_Weight_gpu(exp_Mweight_D + (iorientclass_offset + first_iorient)*exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans,
				                exp_significant_sum_weight_D,
				                isSignificant_D,
				                valid_weight_list_D,
				                exp_Mweight_D_size,
				                exp_nr_images,
				                current_nr_orients,
				                exp_nr_oversampled_rot,
				                exp_nr_trans,
				                exp_nr_oversampled_trans);
				//calculate the prefix sum of the valid weight flags,
				cudaMemcpy(compact_position_list_D, valid_weight_list_D, exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans * sizeof(int), cudaMemcpyDeviceToDevice);
				thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast<int> (compact_position_list_D);

				thrust::exclusive_scan(dev_ptr,
				                       dev_ptr + (exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans),
				                       dev_ptr);
				thrust::device_ptr<int> dev_ptr_sig = thrust::device_pointer_cast<int> (isSignificant_D);
				//calculate the prefix sum of the valid orients, it will be as the output index for the last two kernels
				thrust::exclusive_scan(dev_ptr_sig,
				                       dev_ptr_sig + current_nr_orients,
				                       dev_ptr_sig);
				int num_valid_weight_global;
				int*  weight_index;
				int* image_index_flag_D;
				cudaMemcpy(&num_valid_weight_global, compact_position_list_D + (exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans - 1), sizeof(int), cudaMemcpyDeviceToHost);

				if (num_valid_weight_global > 0)
				{
					cudaMalloc((void**)&exp_local_weight_D, (num_valid_weight_global + 1)*sizeof(DOUBLE));
					cudaMalloc((void**)&weight_index, (num_valid_weight_global + 1)*sizeof(int));
					cudaMalloc((void**)&image_index_flag_D, (num_valid_weight_global + 1) * sizeof(int));
					compact_Weight_gpu(exp_Mweight_D + (iorientclass_offset + first_iorient)*exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans,
					                   exp_significant_sum_weight_D + exp_nr_images,
					                   valid_weight_list_D,
					                   compact_position_list_D,
					                   exp_Mweight_D_size,
					                   exp_nr_images,
					                   current_nr_orients,
					                   exp_nr_oversampled_rot,
					                   exp_nr_trans,
					                   exp_nr_oversampled_trans,
					                   iorientclass_offset,
					                   exp_local_weight_D,
					                   weight_index,
					                   image_index_flag_D
					                  );
					cudaDeviceSynchronize();

					if (!do_skip_maximization)
					{
						calculate_sum_shift_img_gpu(Fimg_D,
						                            fweight_D,
						                            exp_local_Fimgs_shifted_nomask_D,
						                            exp_Minvsigma2s_D,
						                            exp_local_weight_D,
						                            mctf_out_D,
						                            weight_index,
						                            isSignificant_D,
						                            shift_img_size,
						                            num_valid_weight_global,
						                            exp_nr_images,
						                            current_nr_orients,
						                            exp_nr_oversampled_rot,
						                            exp_nr_trans,
						                            exp_nr_oversampled_trans,
						                            current_nr_valid_orients);
						cudaDeviceSynchronize();
						calculate_wdiff2_sumXA_Meta_total_gpu(
						    frefctf_D,
						    exp_local_Fimgs_shifted_D,
						    exp_local_weight_D,
						    weight_index,
						    isSignificant_D,
						    Mresol_fine.nzyxdim,
						    do_scale_correction,
						    num_valid_weight_global,
						    shift_img_size,
						    exp_nr_images,
						    current_nr_orients, //nr_orients
						    exp_nr_oversampled_rot,
						    exp_nr_trans,
						    exp_nr_oversampled_trans,
						    current_nr_valid_orients,//nr_valid_orient,
						    thr_wsum_sigma2_noise_D,
						    thr_wsum_norm_correction_D,
						    thr_wsum_scale_correction_XA_D,
						    thr_wsum_scale_correction_AA_D,
						    data_vs_prior_class_D,
						    mresol_fine_D,
						    group_id_D,
						    (mymodel.ori_size / 2 + 1),
						    mymodel.ref_dim,
						    mymodel.orientational_prior_mode,
						    exp_nr_psi,
						    (mymodel.ref_dim == 2) ? XX(mymodel.prior_offset_class[exp_iclass]) : 0,
						    (mymodel.ref_dim == 2) ? YY(mymodel.prior_offset_class[exp_iclass]) : 0,
						    exp_old_offset_D,
						    oversampled_translations_D,
						    mymodel.orientational_prior_mode != NOPRIOR ? pointer_dir_nonzeroprior_D : NULL,
						    thr_sumw_group_D,
						    thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D,
						    exp_prior_D,
						    thr_wsum_pdf_direction_D
						);

						cudaDeviceSynchronize();

					}
					int num_valid_weight_per_particle[exp_nr_images + 1], max_weight_index[exp_nr_images];
					thrust::device_ptr<int> dev_ptr_weight_flag = thrust::device_pointer_cast<int> (image_index_flag_D);
					thrust::device_ptr<int> dev_ptr_weight_index = thrust::device_pointer_cast<int> (weight_index);
					thrust::device_ptr<DOUBLE> dev_ptr_weight = thrust::device_pointer_cast<DOUBLE> (exp_local_weight_D);
					//thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H[0]=thrust::reduce(dev_ptr_weight, dev_ptr_weight+num_valid_weight_global);
					num_valid_weight_per_particle [0] = 0;
					for (int i = 0; i < exp_nr_images; i++)
					{

						int num = thrust::count(dev_ptr_weight_flag,  dev_ptr_weight_flag + num_valid_weight_global, i);
						num_valid_weight_per_particle[i + 1]  = num_valid_weight_per_particle[i] + num;
						if (num > 0)
						{
							thrust::sort_by_key(dev_ptr_weight + num_valid_weight_per_particle[i], dev_ptr_weight + num_valid_weight_per_particle[i + 1], dev_ptr_weight_index + num_valid_weight_per_particle[i]);

							DOUBLE max_weight;
							cudaMemcpy(&max_weight, exp_local_weight_D + num_valid_weight_per_particle[i + 1] - 1, sizeof(DOUBLE), cudaMemcpyDeviceToHost);
							cudaMemcpy(max_weight_index + i, weight_index + num_valid_weight_per_particle[i + 1] - 1, sizeof(int), cudaMemcpyDeviceToHost);
							if (thr_max_weight[i] < max_weight)
							{
								thr_max_weight[i] = max_weight;

								int iorient = (max_weight_index[i] % (nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans)) / (exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans);
								long int idir = iorient / exp_nr_psi;
								long int ipsi = iorient % exp_nr_psi;

								sampling.getOrientations(idir, ipsi, adaptive_oversampling, oversampled_orientations);
								int iover_rot = (max_weight_index[i] % (exp_nr_oversampled_rot * exp_nr_oversampled_trans)) / (exp_nr_oversampled_trans);
								rot = XX(oversampled_orientations[iover_rot]);
								tilt = YY(oversampled_orientations[iover_rot]);
								psi = ZZ(oversampled_orientations[iover_rot]);
								// Get the Euler matrix
								Euler_angles2matrix(rot, tilt, psi, A);

								// Take tilt-series into account
								A = (exp_R_mic * A).inv();
								Euler_matrix2angles(A.inv(), rot, tilt, psi);
								int itrans = (max_weight_index[i] % (exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans)) / (exp_nr_oversampled_rot * exp_nr_oversampled_trans);
								int iover_trans = (max_weight_index[i] % (exp_nr_oversampled_trans));
								//sampling.getTranslations(itrans, adaptive_oversampling, oversampled_translations);
								//oversampled_translations_H[itrans*exp_nr_oversampled_trans+iover_trans]

								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_ROT) = rot;
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_TILT) = tilt;
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_PSI) = psi;
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_XOFF) = XX(exp_old_offset[i]) +   oversampled_translations_H[itrans * exp_nr_oversampled_trans + iover_trans]; //XX(oversampled_translations[iover_trans]);
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_YOFF) = YY(exp_old_offset[i]) +   oversampled_translations_H[itrans * exp_nr_oversampled_trans + iover_trans + exp_nr_trans * exp_nr_oversampled_trans];
								//YY(oversampled_translations[iover_trans]);
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_CLASS) = (DOUBLE)exp_iclass + 1;
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_PMAX) = thr_max_weight[i];
							}

						}

					}
					f2d_x = shift_img_x;
					f2d_y = shift_img_y;

					int data_XDIM = XSIZE(wsum_model.BPref[exp_iclass].data);
					int data_YXDIM = YXSIZE(wsum_model.BPref[exp_iclass].data);
					data_starty = STARTINGY((wsum_model.BPref[exp_iclass]).data);
					data_startz = STARTINGZ((wsum_model.BPref[exp_iclass]).data);
					if (!do_skip_maximization)
					{
						backproject_gpu(Fimg_D,
						                A_array_D,
						                IS_INV,
						                fweight_D,
						                backproject_data_D + exp_iclass * (wsum_model.BPref[exp_iclass]).data.zyxdim,
						                backproject_Weight_D + exp_iclass * (wsum_model.BPref[exp_iclass]).weight.zyxdim,
						                (wsum_model.BPref[exp_iclass]).padding_factor,
						                (wsum_model.BPref[exp_iclass]).r_max,
						                (wsum_model.BPref[exp_iclass]).r_min_nn,
						                f2d_y,
						                f2d_x,
						                current_nr_valid_orients * exp_nr_oversampled_rot, //nr_A
						                exp_nr_oversampled_rot,
						                image_size,
						                (wsum_model.BPref[exp_iclass]).interpolator,
						                data_starty,
						                data_startz,
						                data_YXDIM,
						                data_XDIM,
						                (wsum_model.BPref[exp_iclass]).ref_dim
						               );
					}
					cudaDeviceSynchronize();
					cudaFree(image_index_flag_D);
					cudaFree(weight_index);
					cudaFree(exp_local_weight_D);
				}

				nr_done_orients += current_nr_valid_orients;
				cudaFree(compact_position_list_D);
				cudaFree(valid_weight_list_D);
				cudaFree(Fimg_D);
				cudaFree(fweight_D);
				cudaFree(mctf_out_D);
				cudaFree(frefctf_D);
			}

		}

		cudaMemcpy(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H, thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D, 4 * sizeof(DOUBLE) , cudaMemcpyDeviceToHost);
		cudaMemcpy(thr_sumw_group_H, thr_sumw_group_D, sizeof(DOUBLE) * mymodel.nr_groups, cudaMemcpyDeviceToHost);
		cudaMemcpy(thr_wsum_pdf_direction_H, thr_wsum_pdf_direction_D, sizeof(DOUBLE) *thr_wsum_pdf_direction[exp_iclass].xdim, cudaMemcpyDeviceToHost);

		thr_wsum_sigma2_offset = thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H[1];
		thr_wsum_pdf_class[exp_iclass] = thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H[0];
		if (mymodel.ref_dim == 2)
		{
			thr_wsum_prior_offsetx_class[exp_iclass] = thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H[2];
			thr_wsum_prior_offsety_class[exp_iclass] = thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H[3];

		}
		for (int i = 0; i < mymodel.nr_groups; i++)
		{
			thr_sumw_group[i] = thr_sumw_group_H[i];
		}
		for (int i = 0; i < thr_wsum_pdf_direction[exp_iclass].xdim; i++)
		{
			thr_wsum_pdf_direction[exp_iclass].data[i] = thr_wsum_pdf_direction_H[i];
		}
		cudaFree(Fref_global_D);
		cudaFree(oversampled_translations_D);
		cudaFree(pointer_dir_nonzeroprior_D);
		cudaFree(thr_sumw_group_D);
		cudaFree(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D);
		cudaFree(thr_wsum_pdf_direction_D);

		free(oversampled_translations_H);
		free(pointer_dir_nonzeroprior_H);
		free(thr_wsum_pdf_direction_H);
		free(thr_sumw_group_H);
		free(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H);
	}
	
	cudaMemcpy(thr_wsum_sigma2_noise_H, thr_wsum_sigma2_noise_D, mymodel.nr_groups * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	cudaMemcpy(thr_wsum_norm_correction_H, thr_wsum_norm_correction_D, exp_nr_particles * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	cudaMemcpy(thr_wsum_scale_correction_XA_H, thr_wsum_scale_correction_XA_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	cudaMemcpy(thr_wsum_scale_correction_AA_H, thr_wsum_scale_correction_AA_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);


	if (!do_skip_maximization)
	{
		if (do_scale_correction)
		{
			for (int n = 0; n < exp_nr_particles; n++)
			{
				for (int i = 0; i < (mymodel.ori_size / 2 + 1); i++)
				{
					exp_wsum_scale_correction_XA[n].data[i] += thr_wsum_scale_correction_XA_H[n * (mymodel.ori_size / 2 + 1) + i];
					exp_wsum_scale_correction_AA[n].data[i] += thr_wsum_scale_correction_AA_H[n * (mymodel.ori_size / 2 + 1) + i];

				}

			}
		}

		for (int n = 0; n < exp_nr_particles; n++)
		{
			exp_wsum_norm_correction[n] += thr_wsum_norm_correction_H[n];
		}
		for (int n = 0; n < mymodel.nr_groups; n++)
		{
			for (int i = 0; i < mymodel.ori_size / 2 + 1; i++)
			{
				wsum_model.sigma2_noise[n].data[i] += thr_wsum_sigma2_noise_H[n * (mymodel.ori_size / 2 + 1) + i];
			}

			wsum_model.sumw_group[n] += thr_sumw_group[n];
		}
		for (int n = 0; n < mymodel.nr_classes; n++)
		{
			wsum_model.pdf_class[n] += thr_wsum_pdf_class[n];

			if (mymodel.ref_dim == 2)
			{
				XX(wsum_model.prior_offset_class[n]) += thr_wsum_prior_offsetx_class[n];
				YY(wsum_model.prior_offset_class[n]) += thr_wsum_prior_offsety_class[n];
			}
#ifdef CHECKSIZES
			if (XSIZE(wsum_model.pdf_direction[n]) != XSIZE(thr_wsum_pdf_direction[n]))
			{
				std::cerr << " XSIZE(wsum_model.pdf_direction[n])= " << XSIZE(wsum_model.pdf_direction[n]) << " XSIZE(thr_wsum_pdf_direction[n])= " << XSIZE(thr_wsum_pdf_direction[n]) << std::endl;
				REPORT_ERROR("XSIZE(wsum_model.pdf_direction[n]) != XSIZE(thr_wsum_pdf_direction[n])");
			}
#endif
			wsum_model.pdf_direction[n] += thr_wsum_pdf_direction[n];
		}
		wsum_model.sigma2_offset += thr_wsum_sigma2_offset;
	} // end if !do_skip_maximization

	// Check max_weight for each particle and set exp_metadata
	for (int n = 0; n < exp_nr_particles; n++)
	{
		// Equal-to because of the series: the nth images in a series will have the same maximum as the first one
		if (thr_max_weight[n] >= exp_max_weight[n])
		{
			// Set max_weight
			exp_max_weight[n] = thr_max_weight[n];

			// Set metadata
			long int my_image_no = exp_starting_image_no[n] + exp_iseries;
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT)  = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_ROT);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT) = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_TILT);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI)  = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_PSI);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF) = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_XOFF);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF) = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_YOFF);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CLASS) = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_CLASS);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PMAX) = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_PMAX);
		}
	}
	cudaFree(myscale_D);
	cudaFree(mctf_D);
	cudaFree(group_id_D);

	cudaFree(thr_wsum_sigma2_noise_D);
	cudaFree(thr_wsum_norm_correction_D);
	cudaFree(thr_wsum_scale_correction_XA_D);
	cudaFree(thr_wsum_scale_correction_AA_D);
	cudaFree(mresol_fine_D);
	cudaFree(data_vs_prior_class_D);

	cudaFree(A_array_D);
	cudaFree(isSignificant_D);
	cudaFree(Significant_list_D);
	cudaFree(exp_old_offset_D);
	cudaFree(exp_prior_D);
	cudaFree(exp_significant_sum_weight_D);

	free(group_id_H);
	//free(oversampled_translations_H);
	free(fimg_H);
	free(fweight_H);
	free(exp_Minvsigma2s_H);
	free(mctf_H);
	free(myscale_H);
	free(A_array_H);
	free(isSignificant_H);
	free(exp_significant_sum_weight_H);
	free(exp_old_offset_H);
	free(exp_prior_H);
	free(Significant_list_H);

	free(thr_wsum_sigma2_noise_H);
	free(thr_wsum_norm_correction_H);
	free(thr_wsum_scale_correction_XA_H);
	free(thr_wsum_scale_correction_AA_H);
}

void MlOptimiser::doThreadStoreWeightedSumsAllOrientations_multi_class_gpu()
{
	int image_size = shift_img_size;//exp_local_Fimgs_shifted[0].zyxdim;
	long int nr_orients = exp_nr_rot;//sampling.NrDirections() * sampling.NrPsiSamplings();
	
	CUFFT_COMPLEX *  frefctf_D, *Fimg_D;
	DOUBLE* mctf_D, *mctf_H, *mctf_out_D;
	DOUBLE* myscale_D, *myscale_H;
	DOUBLE* fweight_D;
	//DOUBLE* fimg_H, *fweight_H;
	DOUBLE* exp_significant_sum_weight_D, *exp_significant_sum_weight_H;
	DOUBLE* exp_local_weight_D;
	int*  group_id_H,  *group_id_D;
	DOUBLE* thr_wsum_sigma2_noise_D, *thr_wsum_norm_correction_D,
	        *thr_wsum_scale_correction_XA_D, *thr_wsum_scale_correction_AA_D;

	DOUBLE* thr_wsum_sigma2_noise_H, *thr_wsum_norm_correction_H,
	        *thr_wsum_scale_correction_XA_H, *thr_wsum_scale_correction_AA_H;
	DOUBLE* data_vs_prior_class_D;
	int* mresol_fine_D;
	DOUBLE* exp_old_offset_D, *exp_old_offset_H;
	DOUBLE* exp_prior_H, *exp_prior_D;
	DOUBLE* exp_Minvsigma2s_H;
	DOUBLE* A_array_D, *A_array_H;
	int* isSignificant_H, *isSignificant_D, *Significant_list_H, *Significant_list_D;

	exp_old_offset_H = (DOUBLE*)malloc(sizeof(DOUBLE) * 2 * exp_nr_images);
	exp_prior_H = (DOUBLE*)malloc(sizeof(DOUBLE) * 2 * exp_nr_images);
	mctf_H = (DOUBLE*) malloc(exp_Fimgs[0].zyxdim * exp_nr_images * sizeof(DOUBLE));
	myscale_H = (DOUBLE*) malloc(exp_nr_images * sizeof(DOUBLE));
	exp_significant_sum_weight_H = (DOUBLE*)malloc(exp_nr_images * 2 * sizeof(DOUBLE));

	group_id_H = (int*)malloc(exp_nr_images * sizeof(int));
	thr_wsum_sigma2_noise_H = (DOUBLE*)malloc(mymodel.nr_groups * (mymodel.ori_size / 2 + 1) * sizeof(DOUBLE));
	thr_wsum_norm_correction_H = (DOUBLE*)malloc(exp_nr_particles * sizeof(DOUBLE));
	thr_wsum_scale_correction_XA_H = (DOUBLE*)malloc(exp_nr_particles * (mymodel.ori_size / 2 + 1) * sizeof(DOUBLE));
	thr_wsum_scale_correction_AA_H = (DOUBLE*)malloc(exp_nr_particles * (mymodel.ori_size / 2 + 1) * sizeof(DOUBLE));
	exp_Minvsigma2s_H = (DOUBLE*) malloc(exp_nr_images * exp_local_Minvsigma2s[0].zyxdim * sizeof(DOUBLE));
	A_array_H = (DOUBLE*)malloc(sizeof(DOUBLE) * nr_orients * exp_nr_oversampled_rot * 9);
	isSignificant_H = (int*) malloc(nr_orients * sizeof(int));
	Significant_list_H = (int*) malloc(nr_orients * sizeof(int));
	cudaMalloc((void**)&mctf_D, exp_Fimgs[0].zyxdim * exp_nr_images * sizeof(DOUBLE));
	cudaMalloc((void**)&myscale_D, exp_nr_images * sizeof(DOUBLE));
	cudaMalloc((void**)&exp_significant_sum_weight_D, exp_nr_images * 2 * sizeof(DOUBLE));
	cudaMalloc((void**)&group_id_D, exp_nr_images * sizeof(int));

	cudaMalloc((void**)&mresol_fine_D, Mresol_fine.zyxdim * sizeof(int));
	cudaMalloc((void**)&thr_wsum_sigma2_noise_D, mymodel.nr_groups * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMalloc((void**)&thr_wsum_norm_correction_D, exp_nr_particles * sizeof(DOUBLE));
	cudaMalloc((void**)&thr_wsum_scale_correction_XA_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMalloc((void**)&thr_wsum_scale_correction_AA_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));	

	cudaMalloc((void**)&exp_old_offset_D, sizeof(DOUBLE) * 2 * exp_nr_images);
	cudaMalloc((void**)&exp_prior_D, sizeof(DOUBLE) * 2 * exp_nr_images);
	cudaMalloc((void**)&isSignificant_D, nr_orients * sizeof(int));
	cudaMalloc((void**)&A_array_D, sizeof(DOUBLE) * nr_orients * exp_nr_oversampled_rot * 9);
	cudaMalloc((void**)&Significant_list_D, nr_orients * sizeof(int));

	std::vector< MultidimArray<DOUBLE> > Mctf_Array;

	Mctf_Array.clear();
	Mctf_Array.resize(exp_nr_particles);
	for (int i = 0; i < exp_nr_particles; i++)
	{
		Mctf_Array[i].resize(exp_Fimgs[0]);
		Mctf_Array[i].initConstant(1.);
	}
	MultidimArray<DOUBLE> Minvsigma2;
	//DOUBLE rot, tilt, psi;
	bool have_warned_small_scale = false;
	for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		// loop over all particles inside this ori_particle
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			// Which number was this image in the combined array of iseries and part_idpart_id
			long int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
			int group_id = mydata.getGroupId(part_id, exp_iseries);
			group_id_H[my_image_no] = group_id;

			exp_old_offset_H[my_image_no] = XX(exp_old_offset[my_image_no]);
			exp_old_offset_H[my_image_no + exp_nr_images] = YY(exp_old_offset[my_image_no]);

			exp_prior_H[my_image_no] = XX(exp_prior[my_image_no]);
			exp_prior_H[my_image_no + exp_nr_images] = YY(exp_prior[my_image_no]);

			if (!do_skip_maximization)
			{
				// Apply CTF to reference projection
				if (do_ctf_correction)
				{
					Mctf_Array[my_image_no]  = exp_local_Fctfs[my_image_no];
				}
				else
				{
					// initialise because there are multiple particles and Mctf gets selfMultiplied for scale_correction
					Mctf_Array[my_image_no] .initConstant(1.);
				}

				memcpy(mctf_H + my_image_no * exp_Fimgs[0].zyxdim, Mctf_Array[my_image_no].data, exp_Fimgs[0].zyxdim * sizeof(DOUBLE));

				if (do_scale_correction)
				{
					// TODO: implemenent B-factor as well...
					DOUBLE myscale = mymodel.scale_correction[group_id];
					if (myscale > 10000.)
					{
						std::cerr << " rlnMicrographScaleCorrection= " << myscale << " group= " << group_id + 1 << " my_image_no= " << my_image_no << std::endl;
						REPORT_ERROR("ERROR: rlnMicrographScaleCorrection is very high. Did you normalize your data?");
					}
					else if (myscale < 0.001)
					{

						if (!have_warned_small_scale)
						{
							std::cout << " WARNING: ignoring group " << group_id + 1 << " with very small or negative scale (" << myscale <<
							          "); Use larger groups for more stable scale estimates." << std::endl;
							have_warned_small_scale = true;
						}
						myscale = 0.001;
					}
					myscale_H[my_image_no] = myscale;
				}
			} // end if !do_skip_maximization
			exp_significant_sum_weight_H[ipart] = exp_significant_weight[ipart];
			exp_significant_sum_weight_H[ipart + exp_nr_images] = exp_sum_weight[ipart];
		}
	}

	for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		// loop over all particles inside this ori_particle
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			// Which number was this image in the combined array of iseries and part_idpart_id
			long int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
			if (do_map)
			{
				memcpy(exp_Minvsigma2s_H + my_image_no * exp_local_Minvsigma2s[0].zyxdim, exp_local_Minvsigma2s[my_image_no].data, exp_local_Minvsigma2s[0].zyxdim * sizeof(DOUBLE));
			}
			else
			{
				memcpy(exp_Minvsigma2s_H + my_image_no * exp_local_Minvsigma2s[0].zyxdim, Minvsigma2.data, exp_local_Minvsigma2s[0].zyxdim * sizeof(DOUBLE));
			}
		}
	}
	cudaMemcpy(exp_old_offset_D, exp_old_offset_H, sizeof(DOUBLE) * 2 * exp_nr_images, cudaMemcpyHostToDevice);
	cudaMemcpy(exp_prior_D, exp_prior_H, sizeof(DOUBLE) * 2 * exp_nr_images, cudaMemcpyHostToDevice);
	cudaMemcpy(mctf_D, mctf_H, exp_nr_images * exp_Fimgs[0].zyxdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cudaMemcpy(myscale_D, myscale_H, exp_nr_images * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cudaMemcpy(group_id_D, group_id_H, exp_nr_images * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(exp_significant_sum_weight_D, exp_significant_sum_weight_H, exp_nr_images * 2 * sizeof(DOUBLE), cudaMemcpyHostToDevice);

	cudaMemcpy(exp_Minvsigma2s_D, exp_Minvsigma2s_H, exp_nr_images * exp_Fimgs[0].yxdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	cudaMemcpy(mresol_fine_D, Mresol_fine.data, Mresol_fine.zyxdim * sizeof(int), cudaMemcpyHostToDevice);
	
	for (exp_iclass = iclass_min; exp_iclass <= iclass_max; exp_iclass++){
	std::vector< Matrix1D<DOUBLE> > oversampled_orientations, oversampled_translations;
	Matrix2D<DOUBLE> A;

	//MultidimArray<DOUBLE> Minvsigma2;
	DOUBLE rot, tilt, psi;
	//bool have_warned_small_scale = false;


	// Initialise Minvsigma2 to all-1 for if !do_map
	Minvsigma2.resize(exp_Fimgs[0]);
	Minvsigma2.initConstant(1.);

	// Make local copies of weighted sums (excepts BPrefs, which are too big)
	// so that there are not too many mutex locks below
	std::vector<MultidimArray<DOUBLE> > thr_wsum_sigma2_noise, thr_wsum_scale_correction_XA, thr_wsum_scale_correction_AA, thr_wsum_pdf_direction;
	std::vector<DOUBLE> thr_wsum_norm_correction, thr_sumw_group, thr_wsum_pdf_class, thr_wsum_prior_offsetx_class, thr_wsum_prior_offsety_class, thr_max_weight;
	DOUBLE thr_wsum_sigma2_offset;
	MultidimArray<DOUBLE> thr_metadata, zeroArray;

	//==========================================================================
	//GPU computing related data structures
	
	int  f2d_x, f2d_y, data_x, data_y, data_z, data_starty, data_startz;

	f2d_x = shift_img_x;
	f2d_y = shift_img_y;

	data_x = XSIZE(mymodel.PPref[exp_iclass].data);
	data_y = YSIZE(mymodel.PPref[exp_iclass].data);
	data_z = ZSIZE(mymodel.PPref[exp_iclass].data);

	data_starty = STARTINGY(mymodel.PPref[exp_iclass].data);
	data_startz = STARTINGZ(mymodel.PPref[exp_iclass].data);
	int nr_A = 0 ;
	
	cudaMalloc((void**)&data_vs_prior_class_D, mymodel.data_vs_prior_class[exp_iclass].zyxdim * sizeof(DOUBLE));

	cudaMemset(thr_wsum_sigma2_noise_D, 0. , mymodel.nr_groups * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMemset(thr_wsum_norm_correction_D, 0. , exp_nr_particles * sizeof(DOUBLE));
	cudaMemset(thr_wsum_scale_correction_XA_D, 0. , exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
	cudaMemset(thr_wsum_scale_correction_AA_D, 0. , exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE));
	//============================================================================
	// Wsum_sigma_noise2 is a 1D-spectrum for each group
	zeroArray.initZeros(mymodel.ori_size / 2 + 1);
	thr_wsum_sigma2_noise.resize(mymodel.nr_groups);
	for (int n = 0; n < mymodel.nr_groups; n++)
	{
		thr_wsum_sigma2_noise[n] = zeroArray;
	}
	// scale-correction terms are a spectrum for each particle
	thr_wsum_scale_correction_XA.resize(exp_nr_particles);
	thr_wsum_scale_correction_AA.resize(exp_nr_particles);
	for (int n = 0; n < exp_nr_particles; n++)
	{
		thr_wsum_scale_correction_XA[n] = zeroArray;
		thr_wsum_scale_correction_AA[n] = zeroArray;
	}
	// wsum_pdf_direction is a 1D-array (of length sampling.NrDirections(0, true)) for each class
	zeroArray.initZeros(sampling.NrDirections(0, true));
	thr_wsum_pdf_direction.resize(mymodel.nr_classes);
	for (int n = 0; n < mymodel.nr_classes; n++)
	{
		thr_wsum_pdf_direction[n] = zeroArray;
	}
	// wsum_norm_correction is a DOUBLE for each particle
	thr_wsum_norm_correction.resize(exp_nr_particles, 0.);
	// sumw_group is a DOUBLE for each group
	thr_sumw_group.resize(mymodel.nr_groups, 0.);
	// wsum_pdf_class is a DOUBLE for each class
	thr_wsum_pdf_class.resize(mymodel.nr_classes, 0.);
	if (mymodel.ref_dim == 2)
	{
		thr_wsum_prior_offsetx_class.resize(mymodel.nr_classes, 0.);
		thr_wsum_prior_offsety_class.resize(mymodel.nr_classes, 0.);
	}
	// max_weight is a DOUBLE for each particle
	thr_max_weight.resize(exp_nr_particles, 0.);
	// wsum_sigma2_offset is just a DOUBLE
	thr_wsum_sigma2_offset = 0.;
	// metadata is a 2D array of nr_particles x METADATA_LINE_LENGTH
	thr_metadata.initZeros(exp_metadata);

	// exp_iclass loop does not always go from 0 to nr_classes!
	long int iorientclass_offset = exp_iclass * exp_nr_rot;
	size_t first_iorient = 0, last_iorient = nr_orients;
	
	
	//================================================================

	
	cudaMemcpy(data_vs_prior_class_D, mymodel.data_vs_prior_class[exp_iclass].data, mymodel.data_vs_prior_class[exp_iclass].zyxdim * sizeof(DOUBLE), cudaMemcpyHostToDevice);
	//================================================================
	for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
	{

		long int iorientclass = iorientclass_offset + iorient;
		isSignificant_H[iorient] = isSignificantAnyParticleAnyTranslation(iorientclass) ? 1 : 0;
		// Only proceed if any of the particles had any significant coarsely sampled translation
		if (isSignificantAnyParticleAnyTranslation(iorientclass))
		{

			long int idir = iorient / exp_nr_psi;
			long int ipsi = iorient % exp_nr_psi;

			// Now get the oversampled (rot, tilt, psi) triplets
			// This will be only the original (rot,tilt,psi) triplet if (adaptive_oversampling==0)
			sampling.getOrientations(idir, ipsi, adaptive_oversampling, oversampled_orientations);

			// Loop over all oversampled orientations (only a single one in the first pass)
			for (long int iover_rot = 0; iover_rot < exp_nr_oversampled_rot; iover_rot++)
			{
				rot = XX(oversampled_orientations[iover_rot]);
				tilt = YY(oversampled_orientations[iover_rot]);
				psi = ZZ(oversampled_orientations[iover_rot]);
				// Get the Euler matrix
				Euler_angles2matrix(rot, tilt, psi, A);

				// Take tilt-series into account
				A = (exp_R_mic * A).inv();
				memcpy(A_array_H + (nr_A + iover_rot) * 9, A.mdata, sizeof(DOUBLE) * 9);
			}
			nr_A += exp_nr_oversampled_rot;
		}
	}
	if (nr_A > 0)
	{

		CUFFT_COMPLEX * Fref_global_D;

		DOUBLE* oversampled_translations_H, *oversampled_translations_D;
		DOUBLE* pointer_dir_nonzeroprior_H, *pointer_dir_nonzeroprior_D;
		DOUBLE* thr_wsum_pdf_direction_D, *thr_wsum_pdf_direction_H;

		DOUBLE* thr_sumw_group_D, *thr_sumw_group_H;
		DOUBLE* thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D;
		int nr_done_orients = 0;
		int current_nr_valid_orients ;
		DOUBLE*  thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H;

		thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H = (DOUBLE*) malloc(4 * sizeof(DOUBLE));
		thr_wsum_pdf_direction_H = (DOUBLE*)malloc(thr_wsum_pdf_direction[exp_iclass].xdim * sizeof(DOUBLE));
		pointer_dir_nonzeroprior_H = (DOUBLE*)malloc(sampling.pointer_dir_nonzeroprior.size() * sizeof(DOUBLE));
		oversampled_translations_H = (DOUBLE*)malloc(exp_nr_trans * exp_nr_oversampled_trans * 2 * sizeof(DOUBLE));
		thr_sumw_group_H = (DOUBLE*) malloc(sizeof(DOUBLE) * mymodel.nr_groups);

		cudaMalloc((void**)&pointer_dir_nonzeroprior_D, sampling.pointer_dir_nonzeroprior.size() * sizeof(DOUBLE));
		cudaMalloc((void**)&thr_wsum_pdf_direction_D, thr_wsum_pdf_direction[exp_iclass].xdim * sizeof(DOUBLE));
		cudaMemset(thr_wsum_pdf_direction_D, 0., thr_wsum_pdf_direction[exp_iclass].xdim * sizeof(DOUBLE));
		cudaMalloc((void**)&oversampled_translations_D, exp_nr_trans * exp_nr_oversampled_trans * 2 * sizeof(DOUBLE));
		cudaMalloc((void**)&thr_sumw_group_D, sizeof(DOUBLE) * mymodel.nr_groups);
		cudaMemset(thr_sumw_group_D, 0., mymodel.nr_groups * sizeof(DOUBLE));
		cudaMalloc((void**)&thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D, 4 * sizeof(DOUBLE));
		cudaMemset(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D, 0., 4 * sizeof(DOUBLE));

		for (long int itrans = 0; itrans < exp_nr_trans; itrans++)
		{
			sampling.getTranslations(itrans, adaptive_oversampling, oversampled_translations);
			for (int i = 0; i < exp_nr_oversampled_trans; i++)
			{
				int offset = itrans * exp_nr_oversampled_trans;
				oversampled_translations_H[ i  + offset] = XX(oversampled_translations[i]);
				oversampled_translations_H[ i  + offset + exp_nr_trans * exp_nr_oversampled_trans] = YY(oversampled_translations[i]);
			}
		}
		cudaMemcpy(oversampled_translations_D, oversampled_translations_H, exp_nr_trans * exp_nr_oversampled_trans * 2 * sizeof(DOUBLE), cudaMemcpyHostToDevice);
		if (mymodel.orientational_prior_mode != NOPRIOR)
		{
			for (int i = 0; i < sampling.pointer_dir_nonzeroprior.size(); i++)
			{
				pointer_dir_nonzeroprior_H[i] = sampling.getDirectionNumberAlsoZeroPrior(i);
			}
			cudaMemcpy(pointer_dir_nonzeroprior_D, pointer_dir_nonzeroprior_H, sampling.pointer_dir_nonzeroprior.size()*sizeof(DOUBLE), cudaMemcpyHostToDevice);

		}

		cudaMemcpy(A_array_D, A_array_H, sizeof(DOUBLE) * nr_A * 9, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&Fref_global_D, nr_A * image_size * sizeof(CUFFT_COMPLEX ));

		cudaMemset(Fref_global_D, 0., nr_A * image_size * sizeof(CUFFT_COMPLEX ));
		get2DFourierTransform_gpu(Fref_global_D,
		                          A_array_D,
		                          project_data_D + exp_iclass * (mymodel.PPref[exp_iclass]).data.zyxdim,
		                          IS_INV,
		                          (mymodel.PPref[exp_iclass]).padding_factor,
		                          (mymodel.PPref[exp_iclass]).r_max,
		                          (mymodel.PPref[exp_iclass]).r_min_nn,
		                          f2d_x,
		                          f2d_y,
		                          data_x,
		                          data_y,
		                          data_z,
		                          data_starty,
		                          data_startz,
		                          nr_A,
		                          mymodel.ref_dim);
		cudaDeviceSynchronize();

		//Calculate the number of loops for all orientations in the current class
		//occording to the memmory requirement and the total memory space of GPU
		int orient_stride = nr_orients;
		int nr_orient_loop = 1;
		//Calculate the maximal memory requirement for the current orient
		float weight_mem_size = (exp_nr_images * sizeof(DOUBLE) * exp_Mweight_D_size / (1024 * 1024));
		float usage_mem = weight_mem_size + nr_A * image_size * sizeof(CUFFT_COMPLEX ) / (1024 * 1024);
		//Calculate the maximal memory requirement for the current orient
		float required_mem = (nr_A * image_size * exp_nr_images * sizeof(DOUBLE) * 3  + nr_A * image_size * sizeof(DOUBLE) * 3 + weight_mem_size) / (1024 * 1024);
		//If the require memory space is largere than 75% of the total GPU device memory
		//We loop all the orients of the corrent class through several steps
		if ((required_mem / (device_mem_G - usage_mem)) > 0.6)
		{

			nr_orient_loop = ROUND(required_mem / (device_mem_G - usage_mem)) + 1;
			orient_stride = (nr_orients + nr_orient_loop - 1) / nr_orient_loop;
			std::cout << "warning: the total global memory  is about " << device_mem_G << " MB and usaged is : " <<  usage_mem << std::endl;
			std::cout << "warning: the maximal memory requirement in the current  orientationsi about" << required_mem << " MB" << "  stride is " << orient_stride <<   std::endl;
		}
		for (int nr_loop = 0; nr_loop < nr_orient_loop; nr_loop++)
		{
			first_iorient = nr_loop * orient_stride;
			last_iorient = ((nr_loop + 1) * orient_stride < nr_orients) ? ((nr_loop + 1) * orient_stride) : nr_orients;
			int current_nr_orients = last_iorient - first_iorient;
			current_nr_valid_orients = 0;
			//int nr_valid_orient_trans = 0;

			for (long int iorient = first_iorient; iorient < last_iorient; iorient++)
			{
				// Only proceed if any of the particles had any significant coarsely sampled translation
				if (isSignificant_H[iorient])
				{
					Significant_list_H[current_nr_valid_orients] = iorient;
					current_nr_valid_orients++;
				}
			}
			//There are valid orients in the current loop
			if (current_nr_valid_orients > 0)
			{

				cudaMalloc((void**)&Fimg_D, current_nr_valid_orients * exp_nr_oversampled_rot * image_size * sizeof(CUFFT_COMPLEX ));
				cudaMalloc((void**)&fweight_D, current_nr_valid_orients * exp_nr_oversampled_rot * image_size * sizeof(DOUBLE));

				cudaMalloc((void**)&frefctf_D, current_nr_valid_orients * exp_nr_oversampled_rot * exp_Fimgs[0].zyxdim * exp_nr_images * sizeof(CUFFT_COMPLEX ));
				cudaMalloc((void**)&mctf_out_D,    current_nr_valid_orients * exp_nr_oversampled_rot * exp_Fimgs[0].zyxdim * exp_nr_images * sizeof(DOUBLE));

				cudaMemset(Fimg_D, 0. , current_nr_valid_orients * exp_nr_oversampled_rot * exp_Fimgs[0].zyxdim * sizeof(CUFFT_COMPLEX ));
				cudaMemset(fweight_D, 0. , current_nr_valid_orients * exp_nr_oversampled_rot * exp_Fimgs[0].zyxdim * sizeof(DOUBLE));

				if (!do_skip_maximization)
				{
					calculate_frefctf_Mctf_gpu(frefctf_D,
					                           Fref_global_D + nr_done_orients * exp_nr_oversampled_rot * image_size,
					                           mctf_D,
					                           mctf_out_D,
					                           myscale_D,
					                           current_nr_valid_orients,
					                           exp_nr_oversampled_rot,
					                           exp_nr_images,
					                           exp_Fimgs[0].zyxdim,
					                           do_ctf_correction && refs_are_ctf_corrected,
					                           do_scale_correction);
				}

				cudaDeviceSynchronize();
				int* valid_weight_list_D, *compact_position_list_D;
				cudaMalloc((void**)&valid_weight_list_D, exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans * sizeof(int));
				cudaMalloc((void**)&compact_position_list_D, exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans * sizeof(int));

				cudaMemset(valid_weight_list_D, 0, exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans * sizeof(int));
				cudaMemcpy(isSignificant_D, isSignificant_H + first_iorient, current_nr_orients * sizeof(int), cudaMemcpyHostToDevice);
				sign_Weight_gpu(exp_Mweight_D + (iorientclass_offset + first_iorient)*exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans,
				                exp_significant_sum_weight_D,
				                isSignificant_D,
				                valid_weight_list_D,
				                exp_Mweight_D_size,
				                exp_nr_images,
				                current_nr_orients,
				                exp_nr_oversampled_rot,
				                exp_nr_trans,
				                exp_nr_oversampled_trans);
				//calculate the prefix sum of the valid weight flags,
				cudaMemcpy(compact_position_list_D, valid_weight_list_D, exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans * sizeof(int), cudaMemcpyDeviceToDevice);
				thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast<int> (compact_position_list_D);

				thrust::exclusive_scan(dev_ptr,
				                       dev_ptr + (exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans),
				                       dev_ptr);
				thrust::device_ptr<int> dev_ptr_sig = thrust::device_pointer_cast<int> (isSignificant_D);
				//calculate the prefix sum of the valid orients, it will be as the output index for the last two kernels
				thrust::exclusive_scan(dev_ptr_sig,
				                       dev_ptr_sig + current_nr_orients,
				                       dev_ptr_sig);
				int num_valid_weight_global;
				int*  weight_index;
				int* image_index_flag_D;
				cudaMemcpy(&num_valid_weight_global, compact_position_list_D + (exp_nr_images * current_nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans - 1), sizeof(int), cudaMemcpyDeviceToHost);

				if (num_valid_weight_global > 0)
				{
					cudaMalloc((void**)&exp_local_weight_D, (num_valid_weight_global + 1)*sizeof(DOUBLE));
					cudaMalloc((void**)&weight_index, (num_valid_weight_global + 1)*sizeof(int));
					cudaMalloc((void**)&image_index_flag_D, (num_valid_weight_global + 1) * sizeof(int));
					compact_Weight_gpu(exp_Mweight_D + (iorientclass_offset + first_iorient)*exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans,
					                   exp_significant_sum_weight_D + exp_nr_images,
					                   valid_weight_list_D,
					                   compact_position_list_D,
					                   exp_Mweight_D_size,
					                   exp_nr_images,
					                   current_nr_orients,
					                   exp_nr_oversampled_rot,
					                   exp_nr_trans,
					                   exp_nr_oversampled_trans,
					                   iorientclass_offset,
					                   exp_local_weight_D,
					                   weight_index,
					                   image_index_flag_D
					                  );
					cudaDeviceSynchronize();

					if (!do_skip_maximization)
					{
						calculate_sum_shift_img_gpu(Fimg_D,
						                            fweight_D,
						                            exp_local_Fimgs_shifted_nomask_D,
						                            exp_Minvsigma2s_D,
						                            exp_local_weight_D,
						                            mctf_out_D,
						                            weight_index,
						                            isSignificant_D,
						                            shift_img_size,
						                            num_valid_weight_global,
						                            exp_nr_images,
						                            current_nr_orients,
						                            exp_nr_oversampled_rot,
						                            exp_nr_trans,
						                            exp_nr_oversampled_trans,
						                            current_nr_valid_orients);
						cudaDeviceSynchronize();
						calculate_wdiff2_sumXA_Meta_total_gpu(
						    frefctf_D,
						    exp_local_Fimgs_shifted_D,
						    exp_local_weight_D,
						    weight_index,
						    isSignificant_D,
						    Mresol_fine.nzyxdim,
						    do_scale_correction,
						    num_valid_weight_global,
						    shift_img_size,
						    exp_nr_images,
						    current_nr_orients, //nr_orients
						    exp_nr_oversampled_rot,
						    exp_nr_trans,
						    exp_nr_oversampled_trans,
						    current_nr_valid_orients,//nr_valid_orient,
						    thr_wsum_sigma2_noise_D,
						    thr_wsum_norm_correction_D,
						    thr_wsum_scale_correction_XA_D,
						    thr_wsum_scale_correction_AA_D,
						    data_vs_prior_class_D,
						    mresol_fine_D,
						    group_id_D,
						    (mymodel.ori_size / 2 + 1),
						    mymodel.ref_dim,
						    mymodel.orientational_prior_mode,
						    exp_nr_psi,
						    (mymodel.ref_dim == 2) ? XX(mymodel.prior_offset_class[exp_iclass]) : 0,
						    (mymodel.ref_dim == 2) ? YY(mymodel.prior_offset_class[exp_iclass]) : 0,
						    exp_old_offset_D,
						    oversampled_translations_D,
						    mymodel.orientational_prior_mode != NOPRIOR ? pointer_dir_nonzeroprior_D : NULL,
						    thr_sumw_group_D,
						    thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D,
						    exp_prior_D,
						    thr_wsum_pdf_direction_D
						);

						cudaDeviceSynchronize();

					}
					int num_valid_weight_per_particle[exp_nr_images + 1], max_weight_index[exp_nr_images];
					thrust::device_ptr<int> dev_ptr_weight_flag = thrust::device_pointer_cast<int> (image_index_flag_D);
					thrust::device_ptr<int> dev_ptr_weight_index = thrust::device_pointer_cast<int> (weight_index);
					thrust::device_ptr<DOUBLE> dev_ptr_weight = thrust::device_pointer_cast<DOUBLE> (exp_local_weight_D);
					//thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H[0]=thrust::reduce(dev_ptr_weight, dev_ptr_weight+num_valid_weight_global);
					num_valid_weight_per_particle [0] = 0;
					for (int i = 0; i < exp_nr_images; i++)
					{

						int num = thrust::count(dev_ptr_weight_flag,  dev_ptr_weight_flag + num_valid_weight_global, i);
						num_valid_weight_per_particle[i + 1]  = num_valid_weight_per_particle[i] + num;
						if (num > 0)
						{
							thrust::sort_by_key(dev_ptr_weight + num_valid_weight_per_particle[i], dev_ptr_weight + num_valid_weight_per_particle[i + 1], dev_ptr_weight_index + num_valid_weight_per_particle[i]);

							DOUBLE max_weight;
							cudaMemcpy(&max_weight, exp_local_weight_D + num_valid_weight_per_particle[i + 1] - 1, sizeof(DOUBLE), cudaMemcpyDeviceToHost);
							cudaMemcpy(max_weight_index + i, weight_index + num_valid_weight_per_particle[i + 1] - 1, sizeof(int), cudaMemcpyDeviceToHost);
							if (thr_max_weight[i] < max_weight)
							{
								thr_max_weight[i] = max_weight;

								int iorient = (max_weight_index[i] % (nr_orients * exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans)) / (exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans);
								long int idir = iorient / exp_nr_psi;
								long int ipsi = iorient % exp_nr_psi;

								sampling.getOrientations(idir, ipsi, adaptive_oversampling, oversampled_orientations);
								int iover_rot = (max_weight_index[i] % (exp_nr_oversampled_rot * exp_nr_oversampled_trans)) / (exp_nr_oversampled_trans);
								rot = XX(oversampled_orientations[iover_rot]);
								tilt = YY(oversampled_orientations[iover_rot]);
								psi = ZZ(oversampled_orientations[iover_rot]);
								// Get the Euler matrix
								Euler_angles2matrix(rot, tilt, psi, A);

								// Take tilt-series into account
								A = (exp_R_mic * A).inv();
								Euler_matrix2angles(A.inv(), rot, tilt, psi);
								int itrans = (max_weight_index[i] % (exp_nr_oversampled_rot * exp_nr_trans * exp_nr_oversampled_trans)) / (exp_nr_oversampled_rot * exp_nr_oversampled_trans);
								int iover_trans = (max_weight_index[i] % (exp_nr_oversampled_trans));
								//sampling.getTranslations(itrans, adaptive_oversampling, oversampled_translations);
								//oversampled_translations_H[itrans*exp_nr_oversampled_trans+iover_trans]

								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_ROT) = rot;
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_TILT) = tilt;
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_PSI) = psi;
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_XOFF) = XX(exp_old_offset[i]) +   oversampled_translations_H[itrans * exp_nr_oversampled_trans + iover_trans]; //XX(oversampled_translations[iover_trans]);
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_YOFF) = YY(exp_old_offset[i]) +   oversampled_translations_H[itrans * exp_nr_oversampled_trans + iover_trans + exp_nr_trans * exp_nr_oversampled_trans];
								//YY(oversampled_translations[iover_trans]);
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_CLASS) = (DOUBLE)exp_iclass + 1;
								DIRECT_A2D_ELEM(thr_metadata, i, METADATA_PMAX) = thr_max_weight[i];
							}

						}

					}
					f2d_x = shift_img_x;
					f2d_y = shift_img_y;

					int data_XDIM = XSIZE(wsum_model.BPref[exp_iclass].data);
					int data_YXDIM = YXSIZE(wsum_model.BPref[exp_iclass].data);
					data_starty = STARTINGY((wsum_model.BPref[exp_iclass]).data);
					data_startz = STARTINGZ((wsum_model.BPref[exp_iclass]).data);
					if (!do_skip_maximization)
					{
						backproject_gpu(Fimg_D,
						                A_array_D,
						                IS_INV,
						                fweight_D,
						                backproject_data_D + exp_iclass * (wsum_model.BPref[exp_iclass]).data.zyxdim,
						                backproject_Weight_D + exp_iclass * (wsum_model.BPref[exp_iclass]).weight.zyxdim,
						                (wsum_model.BPref[exp_iclass]).padding_factor,
						                (wsum_model.BPref[exp_iclass]).r_max,
						                (wsum_model.BPref[exp_iclass]).r_min_nn,
						                f2d_y,
						                f2d_x,
						                current_nr_valid_orients * exp_nr_oversampled_rot, //nr_A
						                exp_nr_oversampled_rot,
						                image_size,
						                (wsum_model.BPref[exp_iclass]).interpolator,
						                data_starty,
						                data_startz,
						                data_YXDIM,
						                data_XDIM,
						                (wsum_model.BPref[exp_iclass]).ref_dim
						               );
					}
					cudaDeviceSynchronize();
					cudaFree(image_index_flag_D);
					cudaFree(weight_index);
					cudaFree(exp_local_weight_D);
				}

				nr_done_orients += current_nr_valid_orients;
				cudaFree(compact_position_list_D);
				cudaFree(valid_weight_list_D);
				cudaFree(Fimg_D);
				cudaFree(fweight_D);
				cudaFree(mctf_out_D);
				cudaFree(frefctf_D);
			}

		}

		cudaMemcpy(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H, thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D, 4 * sizeof(DOUBLE) , cudaMemcpyDeviceToHost);
		cudaMemcpy(thr_sumw_group_H, thr_sumw_group_D, sizeof(DOUBLE) * mymodel.nr_groups, cudaMemcpyDeviceToHost);
		cudaMemcpy(thr_wsum_pdf_direction_H, thr_wsum_pdf_direction_D, sizeof(DOUBLE) *thr_wsum_pdf_direction[exp_iclass].xdim, cudaMemcpyDeviceToHost);

		thr_wsum_sigma2_offset = thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H[1];
		thr_wsum_pdf_class[exp_iclass] = thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H[0];
		if (mymodel.ref_dim == 2)
		{
			thr_wsum_prior_offsetx_class[exp_iclass] = thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H[2];
			thr_wsum_prior_offsety_class[exp_iclass] = thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H[3];

		}
		for (int i = 0; i < mymodel.nr_groups; i++)
		{
			thr_sumw_group[i] = thr_sumw_group_H[i];
		}
		for (int i = 0; i < thr_wsum_pdf_direction[exp_iclass].xdim; i++)
		{
			thr_wsum_pdf_direction[exp_iclass].data[i] = thr_wsum_pdf_direction_H[i];
		}
		cudaFree(Fref_global_D);
		cudaFree(oversampled_translations_D);
		cudaFree(pointer_dir_nonzeroprior_D);
		cudaFree(thr_sumw_group_D);
		cudaFree(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D);
		cudaFree(thr_wsum_pdf_direction_D);

		free(oversampled_translations_H);
		free(pointer_dir_nonzeroprior_H);
		free(thr_wsum_pdf_direction_H);
		free(thr_sumw_group_H);
		free(thr_wsum_pdf_class_sigma2_offset_prior_offsetx_H);
	}
	
	cudaMemcpy(thr_wsum_sigma2_noise_H, thr_wsum_sigma2_noise_D, mymodel.nr_groups * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	cudaMemcpy(thr_wsum_norm_correction_H, thr_wsum_norm_correction_D, exp_nr_particles * sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	cudaMemcpy(thr_wsum_scale_correction_XA_H, thr_wsum_scale_correction_XA_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);
	cudaMemcpy(thr_wsum_scale_correction_AA_H, thr_wsum_scale_correction_AA_D, exp_nr_particles * (mymodel.ori_size / 2 + 1)*sizeof(DOUBLE), cudaMemcpyDeviceToHost);


	if (!do_skip_maximization)
	{
		if (do_scale_correction)
		{
			for (int n = 0; n < exp_nr_particles; n++)
			{
				for (int i = 0; i < (mymodel.ori_size / 2 + 1); i++)
				{
					exp_wsum_scale_correction_XA[n].data[i] += thr_wsum_scale_correction_XA_H[n * (mymodel.ori_size / 2 + 1) + i];
					exp_wsum_scale_correction_AA[n].data[i] += thr_wsum_scale_correction_AA_H[n * (mymodel.ori_size / 2 + 1) + i];

				}

			}
		}

		for (int n = 0; n < exp_nr_particles; n++)
		{
			exp_wsum_norm_correction[n] += thr_wsum_norm_correction_H[n];
		}
		for (int n = 0; n < mymodel.nr_groups; n++)
		{
			for (int i = 0; i < mymodel.ori_size / 2 + 1; i++)
			{
				wsum_model.sigma2_noise[n].data[i] += thr_wsum_sigma2_noise_H[n * (mymodel.ori_size / 2 + 1) + i];
			}

			wsum_model.sumw_group[n] += thr_sumw_group[n];
		}
		for (int n = 0; n < mymodel.nr_classes; n++)
		{
			wsum_model.pdf_class[n] += thr_wsum_pdf_class[n];

			if (mymodel.ref_dim == 2)
			{
				XX(wsum_model.prior_offset_class[n]) += thr_wsum_prior_offsetx_class[n];
				YY(wsum_model.prior_offset_class[n]) += thr_wsum_prior_offsety_class[n];
			}
#ifdef CHECKSIZES
			if (XSIZE(wsum_model.pdf_direction[n]) != XSIZE(thr_wsum_pdf_direction[n]))
			{
				std::cerr << " XSIZE(wsum_model.pdf_direction[n])= " << XSIZE(wsum_model.pdf_direction[n]) << " XSIZE(thr_wsum_pdf_direction[n])= " << XSIZE(thr_wsum_pdf_direction[n]) << std::endl;
				REPORT_ERROR("XSIZE(wsum_model.pdf_direction[n]) != XSIZE(thr_wsum_pdf_direction[n])");
			}
#endif
			wsum_model.pdf_direction[n] += thr_wsum_pdf_direction[n];
		}
		wsum_model.sigma2_offset += thr_wsum_sigma2_offset;
	} // end if !do_skip_maximization

	// Check max_weight for each particle and set exp_metadata
	for (int n = 0; n < exp_nr_particles; n++)
	{
		// Equal-to because of the series: the nth images in a series will have the same maximum as the first one
		if (thr_max_weight[n] >= exp_max_weight[n])
		{
			// Set max_weight
			exp_max_weight[n] = thr_max_weight[n];

			// Set metadata
			long int my_image_no = exp_starting_image_no[n] + exp_iseries;
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT)  = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_ROT);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT) = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_TILT);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI)  = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_PSI);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF) = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_XOFF);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF) = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_YOFF);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CLASS) = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_CLASS);
			DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PMAX) = DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_PMAX);
		}
	}
	cudaFree(data_vs_prior_class_D);

}
	cudaFree(myscale_D);
	cudaFree(mctf_D);
	cudaFree(group_id_D);

	cudaFree(thr_wsum_sigma2_noise_D);
	cudaFree(thr_wsum_norm_correction_D);
	cudaFree(thr_wsum_scale_correction_XA_D);
	cudaFree(thr_wsum_scale_correction_AA_D);
	cudaFree(mresol_fine_D);

	cudaFree(A_array_D);
	cudaFree(isSignificant_D);
	cudaFree(Significant_list_D);
	cudaFree(exp_old_offset_D);
	cudaFree(exp_prior_D);
	cudaFree(exp_significant_sum_weight_D);

	free(group_id_H);
	free(exp_Minvsigma2s_H);
	free(mctf_H);
	free(myscale_H);
	free(A_array_H);
	free(isSignificant_H);
	free(exp_significant_sum_weight_H);
	free(exp_old_offset_H);
	free(exp_prior_H);
	free(Significant_list_H);

	free(thr_wsum_sigma2_noise_H);
	free(thr_wsum_norm_correction_H);
	free(thr_wsum_scale_correction_XA_H);
	free(thr_wsum_scale_correction_AA_H);
}

void MlOptimiser::maximization_gpu()
{

	// First reconstruct the images for each class
	for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
	{
		if (mymodel.pdf_class[iclass] > 0.)
		{
			/*(wsum_model.BPref[iclass]).reconstruct_gpu(mymodel.Iref[iclass], gridding_nr_iter, do_map,
			                                       mymodel.tau2_fudge_factor, mymodel.tau2_class[iclass], mymodel.sigma2_class[iclass],
			                                       mymodel.data_vs_prior_class[iclass], mymodel.fsc_halves_class[iclass], wsum_model.pdf_class[iclass],
			                                       false, false, nr_threads, minres_map);*/
			    if (wsum_model.BPref[iclass].pad_size >512)
			    {
					wsum_model.BPref[iclass].reconstruct(mymodel.Iref[iclass], gridding_nr_iter, do_map,
			                                       mymodel.tau2_fudge_factor, mymodel.tau2_class[iclass], mymodel.sigma2_class[iclass],
			                                       mymodel.data_vs_prior_class[iclass], mymodel.fsc_halves_class[iclass], wsum_model.pdf_class[iclass],
			                                       false, false, nr_threads, minres_map);;

				}
				else
				{
					(wsum_model.BPref[iclass]).reconstruct_gpu(mymodel.Iref[iclass], gridding_nr_iter, do_map,
			                                       mymodel.tau2_fudge_factor, mymodel.tau2_class[iclass], mymodel.sigma2_class[iclass],
			                                       mymodel.data_vs_prior_class[iclass], mymodel.fsc_halves_class[iclass], wsum_model.pdf_class[iclass],
			                                       false, false, nr_threads, minres_map);
			        }
		}
		else
		{
			mymodel.Iref[iclass].initZeros();
		}

	}

	// Then perform the update of all other model parameters
	maximizationOtherParameters();

	// Keep track of changes in hidden variables
	updateOverallChangesInHiddenVariables();

	// This doesn't really work, and I need the original priors for the polishing...
	//if (do_realign_movies)
	//  updatePriorsForMovieFrames();

}

void MlOptimiser::compare_CPU_GPU(Complex* data_cpu, CUFFT_COMPLEX * data_gpu, long int size, char* name, bool is_exit)
{
	Complex* data_gpu_H = (Complex*) malloc(size * sizeof(Complex));
	cudaMemcpy(data_gpu_H, data_gpu, size * sizeof(Complex), cudaMemcpyDeviceToHost);

	for (long int i = 0; i < size; i++)
	{
		if (abs(data_cpu[i] - data_gpu_H[i]) /abs(data_cpu[i])> 1e-4)
		{
			printf("wrong in %s %d %f %f %f %f\n", name, i, data_cpu[i].real, data_cpu[i].imag, data_gpu_H[i].real, data_gpu_H[i].imag);
			if (is_exit)
			{
				exit(EXIT_FAILURE);
			}
		}
	}
	free(data_gpu_H);
}

void MlOptimiser::compare_CPU_GPU(DOUBLE* data_cpu, DOUBLE* data_gpu, long int size, char* name, bool is_exit)
{
	DOUBLE* data_gpu_H = (DOUBLE*) malloc(size * sizeof(DOUBLE));
	cudaMemcpy(data_gpu_H, data_gpu, size * sizeof(DOUBLE), cudaMemcpyDeviceToHost);

	for (long int i = 0; i < size; i++)
	{
		if (abs(data_cpu[i] - data_gpu_H[i]) > 1e-4)
		{
			printf("wrong in %s %d %f %f\n", name, i, data_cpu[i], data_gpu_H[i]);
			if (is_exit)
			{
				exit(EXIT_FAILURE);
			}
		}

	}
	free(data_gpu_H);
}
