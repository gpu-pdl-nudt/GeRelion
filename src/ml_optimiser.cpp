/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
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
  /***************************************************************************
 * Modified by Huayou SU, who adds executation path for GeRelion, 
 * the GPU functions are suffixed with "_gpu"
 ***************************************************************************/
#include "src/ml_optimiser.h"
#include <cuda_profiler_api.h>
#include "src/math_function.h"
#include "src/timer.h"
#include <cuda.h>
#include <cuda_runtime.h>


//#define DEBUG
//#define DEBUG_CHECKSIZES
//#define CHECKSIZES
//Some global threads management variables
Mutex global_mutex, global_mutex2;
Barrier* global_barrier;
ThreadManager* global_ThreadManager;

extern void centerFFT_2_gpu(DOUBLE* in, DOUBLE* out, int nr_images, int dim, int xdim, int ydim, int zdim, bool forward);

// Global functions to work with threads
void globalGetFourierTransformsAndCtfs(ThreadArgument& thArg)
{
	((MlOptimiser*)thArg.workClass)->doThreadGetFourierTransformsAndCtfs(thArg.thread_id);
}

void globalThreadPrecalculateShiftedImagesCtfsAndInvSigma2s(ThreadArgument& thArg)
{
	((MlOptimiser*)thArg.workClass)->doThreadPrecalculateShiftedImagesCtfsAndInvSigma2s(thArg.thread_id);
}

void globalThreadGetSquaredDifferencesAllOrientations(ThreadArgument& thArg)
{
	((MlOptimiser*)thArg.workClass)->doThreadGetSquaredDifferencesAllOrientations(thArg.thread_id);
}

void globalThreadConvertSquaredDifferencesToWeightsAllOrientations(ThreadArgument& thArg)
{
	((MlOptimiser*)thArg.workClass)->doThreadConvertSquaredDifferencesToWeightsAllOrientations(thArg.thread_id);
}

void globalThreadStoreWeightedSumsAllOrientations(ThreadArgument& thArg)
{
	((MlOptimiser*)thArg.workClass)->doThreadStoreWeightedSumsAllOrientations(thArg.thread_id);
}


/** ========================== I/O operations  =========================== */


void MlOptimiser::usage()
{

	parser.writeUsage(std::cerr);
}

void MlOptimiser::read(int argc, char** argv, int rank)
{
	//#define DEBUG_READ

	parser.setCommandLine(argc, argv);

	if (checkParameter(argc, argv, "--template"))                                                                                                                                                                   
	{
	    parser.addSection("Template options");
	    FILE *fp;
	    std::string fn_in = parser.getOption("--template", "template file");
	    printf("fn_in = %s\n", fn_in.c_str());
	    char name[256];
	    fn_in.copy(name, fn_in.length(), 0);
	    fp =fopen(name, "r");
	    printf("file = %s\n", name);
	    char buf[10240];
	    int new_argc = 1;
	    char new_argv[64][256];
	    while (fscanf(fp, "%s\n", buf)!=EOF)
	    {
	        strcpy(new_argv[new_argc], buf);
	        argv[new_argc] = &new_argv[new_argc][0];
	        new_argc++;
	    }
	    argc = new_argc;
	    parser.setCommandLine(argc, argv);
	}  
	if (checkParameter(argc, argv, "--continue"))
	{
		parser.addSection("Continue options");
		FileName fn_in = parser.getOption("--continue", "_optimiser.star file of the iteration after which to continue");
		// Read in previously calculated parameters
		if (fn_in != "")
		{
			read(fn_in, rank);
		}
		// And look for additional command-line options...
		parseContinue(argc, argv);
	}
	else
	{
		// Start a new run from scratch
		parseInitial(argc, argv);
	}
	 


}

void MlOptimiser::parseContinue(int argc, char** argv)
{
#ifdef DEBUG
	std::cerr << "Entering parseContinue" << std::endl;
#endif

	int general_section = parser.addSection("General options");
	// Not all parameters are accessible here...
	FileName fn_out_new = parser.getOption("--o", "Output rootname", "OLD_ctX");
	if (fn_out_new == "OLD_ctX" || fn_out_new == fn_out)
	{
		fn_out += "_ct" + integerToString(iter);
	}
	else
	{
		fn_out = fn_out_new;
	}

	std::string fnt;
	fnt = parser.getOption("--iter", "Maximum number of iterations to perform", "OLD");
	if (fnt != "OLD")
	{
		nr_iter = textToInteger(fnt);
	}

	fnt = parser.getOption("--tau2_fudge", "Regularisation parameter (values higher than 1 give more weight to the data)", "OLD");
	if (fnt != "OLD")
	{
		mymodel.tau2_fudge_factor = textToFloat(fnt);
	}

	// Solvent flattening
	if (parser.checkOption("--flatten_solvent", "Switch on masking on the references?", "OLD"))
	{
		do_solvent = true;
	}

	// Check whether the mask has changed
	fnt = parser.getOption("--solvent_mask", "User-provided mask for the references", "OLD");
	if (fnt != "OLD")
	{
		fn_mask = fnt;
	}

	// Check whether the secondary mask has changed
	fnt = parser.getOption("--solvent_mask2", "User-provided secondary mask", "OLD");
	if (fnt != "OLD")
	{
		fn_mask2 = fnt;
	}

	// Check whether tau2-spectrum has changed
	fnt = parser.getOption("--tau", "STAR file with input tau2-spectrum (to be kept constant)", "OLD");
	if (fnt != "OLD")
	{
		fn_tau = fnt;
	}

	// Check whether particle diameter has changed
	fnt = parser.getOption("--particle_diameter", "Diameter of the circular mask that will be applied to the experimental images (in Angstroms)", "OLD");
	if (fnt != "OLD")
	{
		particle_diameter = textToFloat(fnt);
	}

	// Check whether to join the random halves again
	do_join_random_halves = parser.checkOption("--join_random_halves", "Join previously split random halves again (typically to perform a final reconstruction).");

	// Re-align movie frames
	int movie_section = parser.addSection("Re-align movie frames");

	fn_data_movie = parser.getOption("--realign_movie_frames", "Input STAR file with the movie frames", "");

	// TODO: add this to EMDL_OPTIMISER and read/write of optimiser.star
	nr_frames_per_prior = textToInteger(parser.getOption("--nr_frames_prior", "Number of movie frames to calculate running-average priors", "5"));

	// (integer-) divide running average width by 2 to have the side only
	// TODO: add this to EMDL_OPTIMISER and read/write of optimiser.star
	movie_frame_running_avg_side = textToInteger(parser.getOption("--movie_frames_running_avg", "Number of movie frames in each running average", "3")) / 2;

	// ORIENTATIONS
	int orientations_section = parser.addSection("Orientations");

	fnt = parser.getOption("--oversampling", "Adaptive oversampling order to speed-up calculations (0=no oversampling, 1=2x, 2=4x, etc)", "OLD");
	if (fnt != "OLD")
	{
		adaptive_oversampling = textToInteger(fnt);
	}

	// Check whether angular sampling has changed
	// Do not do this for auto_refine, but make sure to do this when realigning movies!
	if (!do_auto_refine || fn_data_movie != "")
	{
		directions_have_changed = false;
		fnt = parser.getOption("--healpix_order", "Healpix order for the angular sampling rate on the sphere (before oversampling): hp2=15deg, hp3=7.5deg, etc", "OLD");
		if (fnt != "OLD")
		{
			int _order = textToInteger(fnt);
			if (_order != sampling.healpix_order)
			{
				directions_have_changed = true;
				sampling.healpix_order = _order;
			}
		}

		fnt = parser.getOption("--psi_step", "Angular sampling (before oversampling) for the in-plane angle (default=10deg for 2D, hp sampling for 3D)", "OLD");
		if (fnt != "OLD")
		{
			sampling.psi_step = textToFloat(fnt);
		}

		fnt = parser.getOption("--offset_range", "Search range for origin offsets (in pixels)", "OLD");
		if (fnt != "OLD")
		{
			sampling.offset_range = textToFloat(fnt);
		}

		fnt = parser.getOption("--offset_step", "Sampling rate for origin offsets (in pixels)", "OLD");
		if (fnt != "OLD")
		{
			sampling.offset_step = textToFloat(fnt);
		}
	}

	fnt = parser.getOption("--auto_local_healpix_order", "Minimum healpix order (before oversampling) from which auto-refine procedure will use local searches", "OLD");
	if (fnt != "OLD")
	{
		autosampling_hporder_local_searches = textToInteger(fnt);
	}

	// Check whether the prior mode changes
	DOUBLE _sigma_rot, _sigma_tilt, _sigma_psi, _sigma_off;
	int _mode;
	fnt = parser.getOption("--sigma_ang", "Stddev on all three Euler angles for local angular searches (of +/- 3 stddev)", "OLD");
	if (fnt != "OLD")
	{
		mymodel.orientational_prior_mode = PRIOR_ROTTILT_PSI;
		mymodel.sigma2_rot = mymodel.sigma2_tilt = mymodel.sigma2_psi = textToFloat(fnt) * textToFloat(fnt);
	}
	fnt = parser.getOption("--sigma_rot", "Stddev on the first Euler angle for local angular searches (of +/- 3 stddev)", "OLD");
	if (fnt != "OLD")
	{
		mymodel.orientational_prior_mode = PRIOR_ROTTILT_PSI;
		mymodel.sigma2_rot = textToFloat(fnt) * textToFloat(fnt);
	}
	fnt = parser.getOption("--sigma_tilt", "Stddev on the first Euler angle for local angular searches (of +/- 3 stddev)", "OLD");
	if (fnt != "OLD")
	{
		mymodel.orientational_prior_mode = PRIOR_ROTTILT_PSI;
		mymodel.sigma2_tilt = textToFloat(fnt) * textToFloat(fnt);
	}
	fnt = parser.getOption("--sigma_psi", "Stddev on the in-plane angle for local angular searches (of +/- 3 stddev)", "OLD");
	if (fnt != "OLD")
	{
		mymodel.orientational_prior_mode = PRIOR_ROTTILT_PSI;
		mymodel.sigma2_psi = textToFloat(fnt) * textToFloat(fnt);
	}
	fnt = parser.getOption("--sigma_off", "Stddev. on the translations", "OLD");
	if (fnt != "OLD")
	{
		mymodel.sigma2_offset = textToFloat(fnt) * textToFloat(fnt);
	}

	if (parser.checkOption("--skip_align", "Skip orientational assignment (only classify)?"))
	{
		do_skip_align = true;
	}

	if (parser.checkOption("--skip_rotate", "Skip rotational assignment (only translate and classify)?"))
	{
		do_skip_rotate = true;
	}
	else
	{
		do_skip_rotate = false;    // do_skip_rotate should normally be false...
	}

	do_skip_maximization = parser.checkOption("--skip_maximize", "Skip maximization step (only write out data.star file)?");

	int corrections_section = parser.addSection("Corrections");

	// Can only switch the following option ON, not OFF
	if (parser.checkOption("--scale", "Switch on intensity-scale corrections on image groups", "OLD"))
	{
		do_scale_correction = true;
	}

	// Can only switch the following option ON, not OFF
	if (parser.checkOption("--norm", "Switch on normalisation-error correction", "OLD"))
	{
		do_norm_correction = true;
	}

	int computation_section = parser.addSection("Computation");

	nr_threads = textToInteger(parser.getOption("--j", "Number of threads to run in parallel (only useful on multi-core machines)", "1"));

	fnt = parser.getOption("--pool", "Number of images to be processed together", "OLD");
	if (fnt != "OLD")
	{
		max_nr_pool = textToInteger(fnt);
	}

	combine_weights_thru_disc = !parser.checkOption("--dont_combine_weights_via_disc", "Send the large arrays of summed weights through the MPI network, instead of writing large files to disc");

	verb = textToInteger(parser.getOption("--verb", "Verbosity (1=normal, 0=silent)", "1"));

	int expert_section = parser.addSection("Expert options");

	fnt = parser.getOption("--strict_highres_exp", "Resolution limit (in Angstrom) to restrict probability calculations in the expectation step", "OLD");
	if (fnt != "OLD")
	{
		strict_highres_exp = textToFloat(fnt);
	}

	// Debugging/analysis/hidden stuff
	do_map = !checkParameter(argc, argv, "--no_map");
	minres_map = textToInteger(getParameter(argc, argv, "--minres_map", "5"));
	gridding_nr_iter = textToInteger(getParameter(argc, argv, "--gridding_iter", "10"));
	debug1 = textToFloat(getParameter(argc, argv, "--debug1", "0."));
	debug2 = textToFloat(getParameter(argc, argv, "--debug2", "0."));
	do_bfactor = checkParameter(argc, argv, "--bfactor");
	// Read in initial sigmaNoise spectrum
	fn_sigma = getParameter(argc, argv, "--sigma", "");
	sigma2_fudge = textToFloat(getParameter(argc, argv, "--sigma2_fudge", "1."));
	do_acc_currentsize_despite_highres_exp = checkParameter(argc, argv, "--accuracy_current_size");
	do_sequential_halves_recons  = checkParameter(argc, argv, "--sequential_halves_recons");
	do_always_join_random_halves = checkParameter(argc, argv, "--always_join_random_halves");
	do_use_all_data = checkParameter(argc, argv, "--use_all_data");
	do_always_cc  = checkParameter(argc, argv, "--always_cc");

	mode = textToInteger(parser.getOption("--mode", "Choosing the computing mode", "1")); //0 means the CPU computing mode, 1for GPU computing
	fn_yellow_map = parser.getOption("--yellow_map", "yellow map ", "");
	fn_yellow_mask = parser.getOption("--yellow_mask", "???", "");
	fn_red_mask = parser.getOption("--red_mask", "???", "");
	fn_fsc_mask = parser.getOption("--fsc_mask", "???", "");
	mymodel.do_yellow_red_mask = (fn_red_mask != "" && fn_yellow_mask!="" );
	mymodel.do_yellow_map_red_mask = (fn_red_mask != "" && fn_yellow_mask!="" );
	extra_iter = textToInteger(parser.getOption("--extra_iter", "Maximum number of iterations to perform", "0"));
        if(extra_iter !=0)
                nr_iter = iter +extra_iter;
	//sub_extract = true;
	printf("before init sub_extract \n");
	sub_extract = (fn_red_mask != "" && (fn_yellow_mask!="" || fn_yellow_map!=""));
	printf("after init sub_extract \n");
	if (sub_extract)
		printf("true\n");
	else
		printf("false\n");

	do_print_metadata_labels = false;
	do_print_symmetry_ops = false;
#ifdef DEBUG
	std::cerr << "Leaving parseContinue" << std::endl;
#endif

}

void MlOptimiser::parseInitial(int argc, char** argv)
{
#ifdef DEBUG_READ
	std::cerr << "MlOptimiser::parseInitial Entering " << std::endl;
#endif

	// Read/initialise mymodel and sampling from a STAR file
	FileName fn_model = getParameter(argc, argv, "--model", "None");
	if (fn_model != "None")
	{
		mymodel.read(fn_model);
	}
	// Read in the sampling information from a _sampling.star file
	FileName fn_sampling = getParameter(argc, argv, "--sampling", "None");
	if (fn_sampling != "None")
	{
		sampling.read(fn_sampling);
	}

	// General optimiser I/O stuff
	int general_section = parser.addSection("General options");
	fn_data = parser.getOption("--i", "Input images (in a star-file or a stack)");
	fn_out = parser.getOption("--o", "Output rootname");
	nr_iter = textToInteger(parser.getOption("--iter", "Maximum number of iterations to perform", "50"));
	extra_iter = textToInteger(parser.getOption("--extra_iter", "Maximum number of iterations to perform", "0"));
        if(extra_iter!=0)
                nr_iter = iter +extra_iter;
	mymodel.pixel_size = textToFloat(parser.getOption("--angpix", "Pixel size (in Angstroms)"));
	mymodel.tau2_fudge_factor = textToFloat(parser.getOption("--tau2_fudge", "Regularisation parameter (values higher than 1 give more weight to the data)", "1"));
	mymodel.nr_classes = textToInteger(parser.getOption("--K", "Number of references to be refined", "1"));
	particle_diameter = textToFloat(parser.getOption("--particle_diameter", "Diameter of the circular mask that will be applied to the experimental images (in Angstroms)", "-1"));
	do_zero_mask = parser.checkOption("--zero_mask", "Mask surrounding background in particles to zero (by default the solvent area is filled with random noise)");
	do_solvent = parser.checkOption("--flatten_solvent", "Perform masking on the references as well?");
	fn_mask = parser.getOption("--solvent_mask", "User-provided mask for the references (default is to use spherical mask with particle_diameter)", "None");
	fn_mask2 = parser.getOption("--solvent_mask2", "User-provided secondary mask (with its own average density)", "None");
	fn_tau = parser.getOption("--tau", "STAR file with input tau2-spectrum (to be kept constant)", "None");
	do_split_random_halves = parser.checkOption("--split_random_halves", "Refine two random halves of the data completely separately");
	low_resol_join_halves = textToFloat(parser.getOption("--low_resol_join_halves", "Resolution (in Angstrom) up to which the two random half-reconstructions will not be independent to prevent diverging orientations", "-1"));

	// Initialisation
	int init_section = parser.addSection("Initialisation");
	fn_ref = parser.getOption("--ref", "Image, stack or star-file with the reference(s). (Compulsory for 3D refinement!)", "None");
	mymodel.sigma2_offset = textToFloat(parser.getOption("--offset", "Initial estimated stddev for the origin offsets", "3"));
	mymodel.sigma2_offset *= mymodel.sigma2_offset;

	// Perform cross-product comparison at first iteration
	do_firstiter_cc = parser.checkOption("--firstiter_cc", "Perform CC-calculation in the first iteration (use this if references are not on the absolute intensity scale)");
	ini_high = textToFloat(parser.getOption("--ini_high", "Resolution (in Angstroms) to which to limit refinement in the first iteration ", "-1"));

	// Set the orientations
	int orientations_section = parser.addSection("Orientations");
	// Move these to sampling
	adaptive_oversampling = textToInteger(parser.getOption("--oversampling", "Adaptive oversampling order to speed-up calculations (0=no oversampling, 1=2x, 2=4x, etc)", "1"));
	sampling.healpix_order = textToInteger(parser.getOption("--healpix_order", "Healpix order for the angular sampling (before oversampling) on the (3D) sphere: hp2=15deg, hp3=7.5deg, etc", "2"));
	sampling.psi_step = textToFloat(parser.getOption("--psi_step", "Sampling rate (before oversampling) for the in-plane angle (default=10deg for 2D, hp sampling for 3D)", "-1"));
	sampling.limit_tilt = textToFloat(parser.getOption("--limit_tilt", "Limited tilt angle: positive for keeping side views, negative for keeping top views", "-91"));
	sampling.fn_sym = parser.getOption("--sym", "Symmetry group", "c1");
	sampling.offset_range = textToFloat(parser.getOption("--offset_range", "Search range for origin offsets (in pixels)", "6"));
	sampling.offset_step = textToFloat(parser.getOption("--offset_step", "Sampling rate (before oversampling) for origin offsets (in pixels)", "2"));
	sampling.perturbation_factor = textToFloat(parser.getOption("--perturb", "Perturbation factor for the angular sampling (0=no perturb; 0.5=perturb)", "0.5"));
	do_auto_refine = parser.checkOption("--auto_refine", "Perform 3D auto-refine procedure?");
	autosampling_hporder_local_searches = textToInteger(parser.getOption("--auto_local_healpix_order", "Minimum healpix order (before oversampling) from which autosampling procedure will use local searches", "4"));
	parser.setSection(orientations_section);
	DOUBLE _sigma_ang = textToFloat(parser.getOption("--sigma_ang", "Stddev on all three Euler angles for local angular searches (of +/- 3 stddev)", "-1"));
	DOUBLE _sigma_rot = textToFloat(parser.getOption("--sigma_rot", "Stddev on the first Euler angle for local angular searches (of +/- 3 stddev)", "-1"));
	DOUBLE _sigma_tilt = textToFloat(parser.getOption("--sigma_tilt", "Stddev on the second Euler angle for local angular searches (of +/- 3 stddev)", "-1"));
	DOUBLE _sigma_psi = textToFloat(parser.getOption("--sigma_psi", "Stddev on the in-plane angle for local angular searches (of +/- 3 stddev)", "-1"));
	if (_sigma_ang > 0.)
	{
		mymodel.orientational_prior_mode = PRIOR_ROTTILT_PSI;
		// the sigma-values for the orientational prior are in model (and not in sampling) because one might like to estimate them
		// from the data by calculating weighted sums of all angular differences: therefore it needs to be in wsum_model and thus in mymodel.
		mymodel.sigma2_rot = mymodel.sigma2_tilt = mymodel.sigma2_psi = _sigma_ang * _sigma_ang;
	}
	else if (_sigma_rot > 0. || _sigma_tilt > 0. || _sigma_psi > 0.)
	{
		mymodel.orientational_prior_mode = PRIOR_ROTTILT_PSI;
		mymodel.sigma2_rot  = (_sigma_rot > 0.) ? _sigma_rot * _sigma_rot   : 0.;
		mymodel.sigma2_tilt = (_sigma_tilt > 0.) ? _sigma_tilt * _sigma_tilt : 0.;
		mymodel.sigma2_psi  = (_sigma_psi > 0.) ? _sigma_psi * _sigma_psi   : 0.;
	}
	else
	{
		//default
		mymodel.orientational_prior_mode = NOPRIOR;
		mymodel.sigma2_rot = mymodel.sigma2_tilt = mymodel.sigma2_psi = 0.;
	}
	do_skip_align = parser.checkOption("--skip_align", "Skip orientational assignment (only classify)?");
	do_skip_rotate = parser.checkOption("--skip_rotate", "Skip rotational assignment (only translate and classify)?");
	do_skip_maximization = false;

	// CTF, norm, scale, bfactor correction etc.
	int corrections_section = parser.addSection("Corrections");
	do_ctf_correction = parser.checkOption("--ctf", "Perform CTF correction?");
	intact_ctf_first_peak = parser.checkOption("--ctf_intact_first_peak", "Ignore CTFs until their first peak?");
	refs_are_ctf_corrected = parser.checkOption("--ctf_corrected_ref", "Have the input references been CTF-amplitude corrected?");
	ctf_phase_flipped = parser.checkOption("--ctf_phase_flipped", "Have the data been CTF phase-flipped?");
	only_flip_phases = parser.checkOption("--only_flip_phases", "Only perform CTF phase-flipping? (default is full amplitude-correction)");
	do_norm_correction = parser.checkOption("--norm", "Perform normalisation-error correction?");
	do_scale_correction = parser.checkOption("--scale", "Perform intensity-scale corrections on image groups?");

	// Computation stuff
	// The number of threads is always read from the command line
	int computation_section = parser.addSection("Computation");
	nr_threads = textToInteger(parser.getOption("--j", "Number of threads to run in parallel (only useful on multi-core machines)", "1"));
	available_memory = textToFloat(parser.getOption("--memory_per_thread", "Available RAM (in Gb) for each thread", "2"));
	max_nr_pool = textToInteger(parser.getOption("--pool", "Number of images to be processed together", "8"));
	combine_weights_thru_disc = !parser.checkOption("--dont_combine_weights_via_disc", "Send the large arrays of summed weights through the MPI network, instead of writing large files to disc");
	mode = textToInteger(parser.getOption("--mode", "Choosing the computing mode", "1")); //0 means the CPU computing mode, 1for GPU computing
	fn_yellow_map = parser.getOption("--yellow_map", "Yellow  map", "");
	fn_yellow_mask = parser.getOption("--yellow_mask", "Yellow mask map", "");
	fn_red_mask = parser.getOption("--red_mask", "Red mask map", "");
	fn_fsc_mask = parser.getOption("--fsc_mask", "FSC mask map", "");
	
	mymodel.do_yellow_red_mask = (fn_red_mask != "" && fn_yellow_mask!="" );
	mymodel.do_yellow_map_red_mask = (fn_red_mask != "" && fn_yellow_map!="" );
	sub_extract = (fn_red_mask != "" && (fn_yellow_mask!=""||fn_yellow_map!="") );
	
	//mymodel.do_yellow_red_mask = (fn_red_mask != "" && fn_yellow_mask!="" );
	if (sub_extract)
		printf("true\n");
	else
		printf("false\n");
	// Expert options
	int expert_section = parser.addSection("Expert options");
	mymodel.padding_factor = textToInteger(parser.getOption("--pad", "Oversampling factor for the Fourier transforms of the references", "2"));
	mymodel.interpolator = (parser.checkOption("--NN", "Perform nearest-neighbour instead of linear Fourier-space interpolation?")) ? NEAREST_NEIGHBOUR : TRILINEAR;
	mymodel.r_min_nn = textToInteger(parser.getOption("--r_min_nn", "Minimum number of Fourier shells to perform linear Fourier-space interpolation", "10"));
	verb = textToInteger(parser.getOption("--verb", "Verbosity (1=normal, 0=silent)", "1"));
	random_seed = textToInteger(parser.getOption("--random_seed", "Number for the random seed generator", "-1"));
	max_coarse_size = textToInteger(parser.getOption("--coarse_size", "Maximum image size for the first pass of the adaptive sampling approach", "-1"));
	adaptive_fraction = textToFloat(parser.getOption("--adaptive_fraction", "Fraction of the weights to be considered in the first pass of adaptive oversampling ", "0.999"));
	width_mask_edge = textToInteger(parser.getOption("--maskedge", "Width of the soft edge of the spherical mask (in pixels)", "5"));
	fix_sigma_noise = parser.checkOption("--fix_sigma_noise", "Fix the experimental noise spectra?");
	fix_sigma_offset = parser.checkOption("--fix_sigma_offset", "Fix the stddev in the origin offsets?");
	incr_size = textToInteger(parser.getOption("--incr_size", "Number of Fourier shells beyond the current resolution to be included in refinement", "10"));
	do_print_metadata_labels = parser.checkOption("--print_metadata_labels", "Print a table with definitions of all metadata labels, and exit");
	do_print_symmetry_ops = parser.checkOption("--print_symmetry_ops", "Print all symmetry transformation matrices, and exit");
	strict_highres_exp = textToFloat(parser.getOption("--strict_highres_exp", "Resolution limit (in Angstrom) to restrict probability calculations in the expectation step", "-1"));
	dont_raise_norm_error = parser.checkOption("--dont_check_norm", "Skip the check whether the images are normalised correctly");



	// TODO: read/write do_always_cc in optmiser.star file!!!
	// SA-stuff
	do_sim_anneal = parser.checkOption("--sim_anneal", "Perform simulated-annealing to improve overall convergence of random starting models?");
	temp_ini = textToFloat(parser.getOption("--temp_ini", "Initial temperature (K) for simulated annealing", "1000"));
	temp_fin = textToFloat(parser.getOption("--temp_fin", "Initial temperature (K) for simulated annealing", "1"));
	do_always_cc  = parser.checkOption("--always_cc", "Perform CC-calculation in all iterations (useful for faster denovo model generation?)");

	///////////////// Special stuff for first iteration (only accessible via CL, not through readSTAR ////////////////////

	// When reading from the CL: always start at iteration 1
	iter = 0;
	// When starting from CL: always calculate initial sigma_noise
	do_calculate_initial_sigma_noise = true;
	// Start average norm correction at 1!
	mymodel.avg_norm_correction = 1.;
	// Always initialise the PDF of the directions
	directions_have_changed = true;

	// Only reconstruct and join random halves are only available when continuing an old run
	do_join_random_halves = false;

	// For auto-sampling and convergence check
	nr_iter_wo_resol_gain = 0;
	nr_iter_wo_large_hidden_variable_changes = 0;
	current_changes_optimal_classes = 9999999;
	current_changes_optimal_offsets = 999.;
	current_changes_optimal_orientations = 999.;
	smallest_changes_optimal_classes = 9999999;
	smallest_changes_optimal_offsets = 999.;
	smallest_changes_optimal_orientations = 999.;
	acc_rot = acc_trans = 999.;

	best_resol_thus_far = 1. / 999.;
	has_converged = false;
	has_high_fsc_at_limit = false;
	has_large_incr_size_iter_ago = 0;

	// Never realign movies from the start
	do_realign_movies = false;

	// Debugging/analysis/hidden stuff
	do_map = !checkParameter(argc, argv, "--no_map");
	minres_map = textToInteger(getParameter(argc, argv, "--minres_map", "5"));
	do_bfactor = checkParameter(argc, argv, "--bfactor");
	gridding_nr_iter = textToInteger(getParameter(argc, argv, "--gridding_iter", "10"));
	debug1 = textToFloat(getParameter(argc, argv, "--debug1", "0"));
	debug2 = textToFloat(getParameter(argc, argv, "--debug2", "0"));
	// Read in initial sigmaNoise spectrum
	fn_sigma = getParameter(argc, argv, "--sigma", "");
	do_calculate_initial_sigma_noise = (fn_sigma == "") ? true : false;
	sigma2_fudge = textToFloat(getParameter(argc, argv, "--sigma2_fudge", "1"));
	do_acc_currentsize_despite_highres_exp = checkParameter(argc, argv, "--accuracy_current_size");
	do_sequential_halves_recons  = checkParameter(argc, argv, "--sequential_halves_recons");
	do_always_join_random_halves = checkParameter(argc, argv, "--always_join_random_halves");
	do_use_all_data = checkParameter(argc, argv, "--use_all_data");

#ifdef DEBUG_READ
	std::cerr << "MlOptimiser::parseInitial Done" << std::endl;
#endif

}


void MlOptimiser::read(FileName fn_in, int rank)
{

#ifdef DEBUG_READ
	std::cerr << "MlOptimiser::readStar entering ..." << std::endl;
#endif

	// Open input file
	std::ifstream in(fn_in.data(), std::ios_base::in);
	if (in.fail())
	{
		REPORT_ERROR((std::string) "MlOptimiser::readStar: File " + fn_in + " cannot be read.");
	}

	MetaDataTable MD;

	// Read general stuff
	FileName fn_model, fn_model2, fn_sampling;
	MD.readStar(in, "optimiser_general");
	in.close();

	if (!MD.getValue(EMDL_OPTIMISER_OUTPUT_ROOTNAME, fn_out) ||
	        !MD.getValue(EMDL_OPTIMISER_MODEL_STARFILE, fn_model) ||
	        !MD.getValue(EMDL_OPTIMISER_DATA_STARFILE, fn_data) ||
	        !MD.getValue(EMDL_OPTIMISER_SAMPLING_STARFILE, fn_sampling) ||
	        !MD.getValue(EMDL_OPTIMISER_ITERATION_NO, iter) ||
	        !MD.getValue(EMDL_OPTIMISER_NR_ITERATIONS, nr_iter) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_SPLIT_RANDOM_HALVES, do_split_random_halves) ||
	        !MD.getValue(EMDL_OPTIMISER_LOWRES_JOIN_RANDOM_HALVES, low_resol_join_halves) ||
	        !MD.getValue(EMDL_OPTIMISER_ADAPTIVE_OVERSAMPLING, adaptive_oversampling) ||
	        !MD.getValue(EMDL_OPTIMISER_ADAPTIVE_FRACTION, adaptive_fraction) ||
	        !MD.getValue(EMDL_OPTIMISER_RANDOM_SEED, random_seed) ||
	        !MD.getValue(EMDL_OPTIMISER_PARTICLE_DIAMETER, particle_diameter) ||
	        !MD.getValue(EMDL_OPTIMISER_WIDTH_MASK_EDGE, width_mask_edge) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_ZERO_MASK, do_zero_mask) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_SOLVENT_FLATTEN, do_solvent) ||
	        !MD.getValue(EMDL_OPTIMISER_SOLVENT_MASK_NAME, fn_mask) ||
	        !MD.getValue(EMDL_OPTIMISER_SOLVENT_MASK2_NAME, fn_mask2) ||
	        !MD.getValue(EMDL_OPTIMISER_TAU_SPECTRUM_NAME, fn_tau) ||
	        !MD.getValue(EMDL_OPTIMISER_COARSE_SIZE, coarse_size) ||
	        !MD.getValue(EMDL_OPTIMISER_MAX_COARSE_SIZE, max_coarse_size) ||
	        !MD.getValue(EMDL_OPTIMISER_HIGHRES_LIMIT_EXP, strict_highres_exp) ||
	        !MD.getValue(EMDL_OPTIMISER_INCR_SIZE, incr_size) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_MAP, do_map) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_AUTO_REFINE, do_auto_refine) ||
	        !MD.getValue(EMDL_OPTIMISER_AUTO_LOCAL_HP_ORDER, autosampling_hporder_local_searches) ||
	        !MD.getValue(EMDL_OPTIMISER_NR_ITER_WO_RESOL_GAIN, nr_iter_wo_resol_gain) ||
	        !MD.getValue(EMDL_OPTIMISER_BEST_RESOL_THUS_FAR, best_resol_thus_far) ||
	        !MD.getValue(EMDL_OPTIMISER_NR_ITER_WO_HIDDEN_VAR_CHANGES, nr_iter_wo_large_hidden_variable_changes) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_SKIP_ALIGN, do_skip_align) ||
	        //!MD.getValue(EMDL_OPTIMISER_DO_SKIP_ROTATE, do_skip_rotate) ||
	        !MD.getValue(EMDL_OPTIMISER_ACCURACY_ROT, acc_rot) ||
	        !MD.getValue(EMDL_OPTIMISER_ACCURACY_TRANS, acc_trans) ||
	        !MD.getValue(EMDL_OPTIMISER_CHANGES_OPTIMAL_ORIENTS, current_changes_optimal_orientations) ||
	        !MD.getValue(EMDL_OPTIMISER_CHANGES_OPTIMAL_OFFSETS, current_changes_optimal_offsets) ||
	        !MD.getValue(EMDL_OPTIMISER_CHANGES_OPTIMAL_CLASSES, current_changes_optimal_classes) ||
	        !MD.getValue(EMDL_OPTIMISER_SMALLEST_CHANGES_OPT_ORIENTS, smallest_changes_optimal_orientations) ||
	        !MD.getValue(EMDL_OPTIMISER_SMALLEST_CHANGES_OPT_OFFSETS, smallest_changes_optimal_offsets) ||
	        !MD.getValue(EMDL_OPTIMISER_SMALLEST_CHANGES_OPT_CLASSES, smallest_changes_optimal_classes) ||
	        !MD.getValue(EMDL_OPTIMISER_HAS_CONVERGED, has_converged) ||
	        !MD.getValue(EMDL_OPTIMISER_HAS_HIGH_FSC_AT_LIMIT, has_high_fsc_at_limit) ||
	        !MD.getValue(EMDL_OPTIMISER_HAS_LARGE_INCR_SIZE_ITER_AGO, has_large_incr_size_iter_ago) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_CORRECT_NORM, do_norm_correction) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_CORRECT_SCALE, do_scale_correction) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_CORRECT_CTF, do_ctf_correction) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_REALIGN_MOVIES, do_realign_movies) ||
	        !MD.getValue(EMDL_OPTIMISER_IGNORE_CTF_UNTIL_FIRST_PEAK, intact_ctf_first_peak) ||
	        !MD.getValue(EMDL_OPTIMISER_DATA_ARE_CTF_PHASE_FLIPPED, ctf_phase_flipped) ||
	        !MD.getValue(EMDL_OPTIMISER_DO_ONLY_FLIP_CTF_PHASES, only_flip_phases) ||
	        !MD.getValue(EMDL_OPTIMISER_REFS_ARE_CTF_CORRECTED, refs_are_ctf_corrected) ||
	        !MD.getValue(EMDL_OPTIMISER_FIX_SIGMA_NOISE, fix_sigma_noise) ||
	        !MD.getValue(EMDL_OPTIMISER_FIX_SIGMA_OFFSET, fix_sigma_offset) ||
	        !MD.getValue(EMDL_OPTIMISER_MAX_NR_POOL, max_nr_pool) ||
	        !MD.getValue(EMDL_OPTIMISER_AVAILABLE_MEMORY, available_memory))
	{
		REPORT_ERROR("MlOptimiser::readStar: incorrect optimiser_general table");
	}

	if (do_split_random_halves &&
	        !MD.getValue(EMDL_OPTIMISER_MODEL_STARFILE2, fn_model2))
	{
		REPORT_ERROR("MlOptimiser::readStar: splitting data into two random halves, but rlnModelStarFile2 not found in optimiser_general table");
	}

	// Initialise some stuff for first-iteration only (not relevant here...)
	do_calculate_initial_sigma_noise = false;
	do_average_unaligned = false;
	do_generate_seeds = false;
	do_firstiter_cc = false;
	ini_high = 0;

	// Initialise some of the other, hidden or debugging stuff
	minres_map = 5;
	do_bfactor = false;
	gridding_nr_iter = 10;
	debug1 = debug2 = 0.;

	// Then read in sampling, mydata and mymodel stuff
	mydata.read(fn_data);
	if (do_split_random_halves)
	{
		if (rank % 2 == 1)
		{
			mymodel.read(fn_model);
		}
		else
		{
			mymodel.read(fn_model2);
		}
	}
	else
	{
		mymodel.read(fn_model);
	}
	sampling.read(fn_sampling);

#ifdef DEBUG_READ
	std::cerr << "MlOptimiser::readStar done." << std::endl;
#endif

}


void MlOptimiser::write(bool do_write_sampling, bool do_write_data, bool do_write_optimiser, bool do_write_model, int random_subset)
{

	FileName fn_root, fn_tmp, fn_model, fn_model2, fn_data, fn_sampling;
	std::ofstream  fh;
	if (iter > -1)
	{
		fn_root.compose(fn_out + "_it", iter, "", 3);
	}
	else
	{
		fn_root = fn_out;
	}

	// First write "main" STAR file with all information from this run
	// Do this for random_subset==0 and random_subset==1
	if (do_write_optimiser && random_subset < 2)
	{
		fn_tmp = fn_root + "_optimiser.star";
		fh.open((fn_tmp).c_str(), std::ios::out);
		if (!fh)
		{
			REPORT_ERROR((std::string)"MlOptimiser::write: Cannot write file: " + fn_tmp);
		}

		// Write the command line as a comment in the header
		fh << "# RELION optimiser" << std::endl;
		fh << "# ";
		parser.writeCommandLine(fh);

		if (do_split_random_halves && !do_join_random_halves)
		{
			fn_model  = fn_root + "_half1_model.star";
			fn_model2 = fn_root + "_half2_model.star";
		}
		else
		{
			fn_model = fn_root + "_model.star";
		}
		fn_data = fn_root + "_data.star";
		fn_sampling = fn_root + "_sampling.star";

		MetaDataTable MD;
		MD.setIsList(true);
		MD.setName("optimiser_general");
		MD.addObject();
		MD.setValue(EMDL_OPTIMISER_OUTPUT_ROOTNAME, fn_out);
		if (do_split_random_halves)
		{
			MD.setValue(EMDL_OPTIMISER_MODEL_STARFILE, fn_model);
			MD.setValue(EMDL_OPTIMISER_MODEL_STARFILE2, fn_model2);
		}
		else
		{
			MD.setValue(EMDL_OPTIMISER_MODEL_STARFILE, fn_model);
		}
		MD.setValue(EMDL_OPTIMISER_DATA_STARFILE, fn_data);
		MD.setValue(EMDL_OPTIMISER_SAMPLING_STARFILE, fn_sampling);
		MD.setValue(EMDL_OPTIMISER_ITERATION_NO, iter);
		MD.setValue(EMDL_OPTIMISER_NR_ITERATIONS, nr_iter);
		MD.setValue(EMDL_OPTIMISER_DO_SPLIT_RANDOM_HALVES, do_split_random_halves);
		MD.setValue(EMDL_OPTIMISER_LOWRES_JOIN_RANDOM_HALVES, low_resol_join_halves);
		MD.setValue(EMDL_OPTIMISER_ADAPTIVE_OVERSAMPLING, adaptive_oversampling);
		MD.setValue(EMDL_OPTIMISER_ADAPTIVE_FRACTION, adaptive_fraction);
		MD.setValue(EMDL_OPTIMISER_RANDOM_SEED, random_seed);
		MD.setValue(EMDL_OPTIMISER_PARTICLE_DIAMETER, particle_diameter);
		MD.setValue(EMDL_OPTIMISER_WIDTH_MASK_EDGE, width_mask_edge);
		MD.setValue(EMDL_OPTIMISER_DO_ZERO_MASK, do_zero_mask);
		MD.setValue(EMDL_OPTIMISER_DO_SOLVENT_FLATTEN, do_solvent);
		MD.setValue(EMDL_OPTIMISER_SOLVENT_MASK_NAME, fn_mask);
		MD.setValue(EMDL_OPTIMISER_SOLVENT_MASK2_NAME, fn_mask2);
		MD.setValue(EMDL_OPTIMISER_TAU_SPECTRUM_NAME, fn_tau);
		MD.setValue(EMDL_OPTIMISER_COARSE_SIZE, coarse_size);
		MD.setValue(EMDL_OPTIMISER_MAX_COARSE_SIZE, max_coarse_size);
		MD.setValue(EMDL_OPTIMISER_HIGHRES_LIMIT_EXP, strict_highres_exp);
		MD.setValue(EMDL_OPTIMISER_INCR_SIZE, incr_size);
		MD.setValue(EMDL_OPTIMISER_DO_MAP, do_map);
		MD.setValue(EMDL_OPTIMISER_DO_AUTO_REFINE, do_auto_refine);
		MD.setValue(EMDL_OPTIMISER_AUTO_LOCAL_HP_ORDER, autosampling_hporder_local_searches);
		MD.setValue(EMDL_OPTIMISER_NR_ITER_WO_RESOL_GAIN, nr_iter_wo_resol_gain);
		MD.setValue(EMDL_OPTIMISER_BEST_RESOL_THUS_FAR, best_resol_thus_far);
		MD.setValue(EMDL_OPTIMISER_NR_ITER_WO_HIDDEN_VAR_CHANGES, nr_iter_wo_large_hidden_variable_changes);
		MD.setValue(EMDL_OPTIMISER_DO_SKIP_ALIGN, do_skip_align);
		MD.setValue(EMDL_OPTIMISER_DO_SKIP_ROTATE, do_skip_rotate);
		MD.setValue(EMDL_OPTIMISER_ACCURACY_ROT, acc_rot);
		MD.setValue(EMDL_OPTIMISER_ACCURACY_TRANS, acc_trans);
		MD.setValue(EMDL_OPTIMISER_CHANGES_OPTIMAL_ORIENTS, current_changes_optimal_orientations);
		MD.setValue(EMDL_OPTIMISER_CHANGES_OPTIMAL_OFFSETS, current_changes_optimal_offsets);
		MD.setValue(EMDL_OPTIMISER_CHANGES_OPTIMAL_CLASSES, current_changes_optimal_classes);
		MD.setValue(EMDL_OPTIMISER_SMALLEST_CHANGES_OPT_ORIENTS, smallest_changes_optimal_orientations);
		MD.setValue(EMDL_OPTIMISER_SMALLEST_CHANGES_OPT_OFFSETS, smallest_changes_optimal_offsets);
		MD.setValue(EMDL_OPTIMISER_SMALLEST_CHANGES_OPT_CLASSES, smallest_changes_optimal_classes);
		MD.setValue(EMDL_OPTIMISER_HAS_CONVERGED, has_converged);
		MD.setValue(EMDL_OPTIMISER_HAS_HIGH_FSC_AT_LIMIT, has_high_fsc_at_limit);
		MD.setValue(EMDL_OPTIMISER_HAS_LARGE_INCR_SIZE_ITER_AGO, has_large_incr_size_iter_ago);
		MD.setValue(EMDL_OPTIMISER_DO_CORRECT_NORM, do_norm_correction);
		MD.setValue(EMDL_OPTIMISER_DO_CORRECT_SCALE, do_scale_correction);
		MD.setValue(EMDL_OPTIMISER_DO_CORRECT_CTF, do_ctf_correction);
		MD.setValue(EMDL_OPTIMISER_DO_REALIGN_MOVIES, do_realign_movies);
		MD.setValue(EMDL_OPTIMISER_IGNORE_CTF_UNTIL_FIRST_PEAK, intact_ctf_first_peak);
		MD.setValue(EMDL_OPTIMISER_DATA_ARE_CTF_PHASE_FLIPPED, ctf_phase_flipped);
		MD.setValue(EMDL_OPTIMISER_DO_ONLY_FLIP_CTF_PHASES, only_flip_phases);
		MD.setValue(EMDL_OPTIMISER_REFS_ARE_CTF_CORRECTED, refs_are_ctf_corrected);
		MD.setValue(EMDL_OPTIMISER_FIX_SIGMA_NOISE, fix_sigma_noise);
		MD.setValue(EMDL_OPTIMISER_FIX_SIGMA_OFFSET, fix_sigma_offset);
		MD.setValue(EMDL_OPTIMISER_MAX_NR_POOL, max_nr_pool);
		MD.setValue(EMDL_OPTIMISER_AVAILABLE_MEMORY, available_memory);

		MD.write(fh);
		fh.close();
	}

	// Then write the mymodel to file
	if (do_write_model)
	{
		if (do_split_random_halves && !do_join_random_halves)
		{
			mymodel.write(fn_root + "_half" + integerToString(random_subset), sampling);
		}
		else
		{
			mymodel.write(fn_root, sampling);
		}
	}

	// And write the mydata to file
	if (do_write_data)
	{
		mydata.write(fn_root);
	}

	// And write the sampling object
	if (do_write_sampling)
	{
		sampling.write(fn_root);
	}

}

/** ========================== Initialisation  =========================== */

void MlOptimiser::initialise()
{
#ifdef DEBUG
	std::cerr << "MlOptimiser::initialise Entering" << std::endl;
#endif

	initialiseGeneral();

	initialiseWorkLoad();

	if (fn_sigma != "")
	{
		// Read in sigma_noise spetrum from file DEVELOPMENTAL!!! FOR DEBUGGING ONLY....
		MetaDataTable MDsigma;
		DOUBLE val;
		int idx;
		MDsigma.read(fn_sigma);
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDsigma)
		{
			MDsigma.getValue(EMDL_SPECTRAL_IDX, idx);
			MDsigma.getValue(EMDL_MLMODEL_SIGMA2_NOISE, val);
			if (idx < XSIZE(mymodel.sigma2_noise[0]))
			{
				mymodel.sigma2_noise[0](idx) = val;
			}
		}
		if (idx < XSIZE(mymodel.sigma2_noise[0]) - 1)
		{
			if (verb > 0)
			{
				std::cout << " WARNING: provided sigma2_noise-spectrum has fewer entries (" << idx + 1 << ") than needed (" << XSIZE(mymodel.sigma2_noise[0]) << "). Set rest to zero..." << std::endl;
			}
		}
		// Use the same spectrum for all classes
		for (int igroup = 0; igroup < mymodel.nr_groups; igroup++)
		{
			mymodel.sigma2_noise[igroup] =  mymodel.sigma2_noise[0];
		}

	}
	else if (do_calculate_initial_sigma_noise || do_average_unaligned)
	{
		MultidimArray<DOUBLE> Mavg;

		// Calculate initial sigma noise model from power_class spectra of the individual images
		calculateSumOfPowerSpectraAndAverageImage(Mavg);

		// Set sigma2_noise and Iref from averaged poser spectra and Mavg
		setSigmaNoiseEstimatesAndSetAverageImage(Mavg);
	}

	// First low-pass filter the initial references
	if (iter == 0)
	{
		initialLowPassFilterReferences();
	}

	// Initialise the data_versus_prior ratio to get the initial current_size right
	if (iter == 0)
	{
		mymodel.initialiseDataVersusPrior(fix_tau);    // fix_tau was set in initialiseGeneral
	}

	// Check minimum group size of 10 particles
	if (verb > 0)
	{
		bool do_warn = false;
		for (int igroup = 0; igroup < mymodel.nr_groups; igroup++)
		{
			if (mymodel.nr_particles_group[igroup] < 10)
			{
				std:: cout << "WARNING: There are only " << mymodel.nr_particles_group[igroup] << " particles in group " << igroup + 1 << std::endl;
				do_warn = true;
			}
		}
		if (do_warn)
		{
			std:: cout << "WARNING: You may want to consider joining some micrographs into larger groups to obtain more robust noise estimates. " << std::endl;
			std:: cout << "         You can do so by using the same rlnMicrographName label for particles from multiple different micrographs in the input STAR file. " << std::endl;
		}
	}

	// Write out initial mymodel
	write(DONT_WRITE_SAMPLING, DO_WRITE_DATA, DO_WRITE_OPTIMISER, DO_WRITE_MODEL, 0);


	// Do this after writing out the model, so that still the random halves are written in separate files.
	if (do_realign_movies)
	{
		// Resolution seems to decrease again after 1 iteration. Therefore, just perform a single iteration until we figure out what exactly happens here...
		has_converged = true;
		// Then use join random halves
		do_join_random_halves = true;

		// If we skip the maximization step, then there is no use in using all data
		if (!do_skip_maximization)
		{
			// Use all data out to Nyquist because resolution gains may be substantial
			do_use_all_data = true;
		}
	}

#ifdef DEBUG
	std::cerr << "MlOptimiser::initialise Done" << std::endl;
#endif
}

void MlOptimiser::initialiseGeneral(int rank)
{

#ifdef DEBUG
	std::cerr << "Entering initialiseGeneral" << std::endl;
#endif

#ifdef TIMING
	//DIFFF = timer.setNew("difff");
	TIMING_EXP =           timer.setNew("expectation");
	TIMING_MAX =           timer.setNew("maximization");
	TIMING_RECONS =        timer.setNew("reconstruction");
	TIMING_ESP =           timer.setNew("expectationSomeParticles");
	TIMING_ESP_READ  =     timer.setNew(" - ESP: read");
	TIMING_ESP_DIFF1 =     timer.setNew(" - ESP: getAllSquaredDifferences1");
	TIMING_ESP_DIFF2 =     timer.setNew(" - ESP: getAllSquaredDifferences2");
	TIMING_DIFF_PROJ =     timer.setNew(" -  - ESPdiff2: project");
	TIMING_DIFF_SHIFT =    timer.setNew(" -  - ESPdiff2: shift");
	TIMING_DIFF_DIFF2 =    timer.setNew(" -  - ESPdiff2: diff2");
	TIMING_ESP_WEIGHT1 =   timer.setNew(" - ESP: convertDiff2ToWeights1");
	TIMING_ESP_WEIGHT2 =   timer.setNew(" - ESP: convertDiff2ToWeights2");
	TIMING_WEIGHT_EXP =    timer.setNew(" -  - ESPweight: exp");
	TIMING_WEIGHT_SORT =   timer.setNew(" -  - ESPweight: sort");
	TIMING_ESP_WSUM =      timer.setNew(" - ESP: storeWeightedSums");
	TIMING_WSUM_PROJ =     timer.setNew("  - - ESPwsum: project");
	TIMING_WSUM_DIFF2 =    timer.setNew(" -  - ESPwsum: diff2");
	TIMING_WSUM_SUMSHIFT = timer.setNew(" -  - ESPwsum: shift");
	TIMING_WSUM_BACKPROJ = timer.setNew(" -  - ESPwsum: backproject");
#endif
#ifdef FLOAT_PRECISION
        if (verb > 0)
            std::cout << " Running in single precision. Runs might not be exactly reproducible." << std::endl;
#else
        if (verb > 0)
            std::cout << " Running in double precision. " << std::endl;
#endif
	if (do_print_metadata_labels)
	{
		if (verb > 0)
		{
			EMDL::printDefinitions(std::cout);
		}
		exit(0);
	}

	// Print symmetry operators to cout
	if (do_print_symmetry_ops)
	{
		if (verb > 0)
		{
			SymList SL;
			SL.writeDefinition(std::cout, sampling.symmetryGroup());
		}
		exit(0);
	}

	// Check for errors in the command-line option
	if (parser.checkForErrors(verb))
	{
		REPORT_ERROR("Errors encountered on the command line (see above), exiting...");
	}

	nr_threads_original = nr_threads;
	// If we are not continuing an old run, now read in the data and the reference images
	if (iter == 0)
	{

		// Read in the experimental image metadata
		mydata.read(fn_data);

		// Also get original size of the images to pass to mymodel.read()
		int ori_size = -1;
		mydata.MDexp.getValue(EMDL_IMAGE_SIZE, ori_size);
		if (ori_size % 2 != 0)
		{
			REPORT_ERROR("This program only works with even values for the image dimensions!");
		}
		mymodel.readImages(fn_ref, ori_size, mydata,
		                   do_average_unaligned, do_generate_seeds, refs_are_ctf_corrected);

		// Check consistency of EMDL_CTF_MAGNIFICATION and MEBL_CTF_DETECTOR_PIXEL_SIZE with mymodel.pixel_size
		DOUBLE mag, dstep, first_angpix, my_angpix;
		bool has_magn = false;
		if (mydata.MDimg.containsLabel(EMDL_CTF_MAGNIFICATION) && mydata.MDimg.containsLabel(EMDL_CTF_DETECTOR_PIXEL_SIZE))
		{
			FOR_ALL_OBJECTS_IN_METADATA_TABLE(mydata.MDimg)
			{
				mydata.MDimg.getValue(EMDL_CTF_MAGNIFICATION, mag);
				mydata.MDimg.getValue(EMDL_CTF_DETECTOR_PIXEL_SIZE, dstep);
				my_angpix = 10000. * dstep / mag;
				if (!has_magn)
				{
					first_angpix = my_angpix;
					has_magn = true;
				}
				else if (ABS(first_angpix - my_angpix) > 0.01)
				{
					REPORT_ERROR("MlOptimiser::initialiseGeneral: ERROR inconsistent magnification and detector pixel sizes in images in input STAR file");
				}
			}
		}
		if (mydata.MDmic.containsLabel(EMDL_CTF_MAGNIFICATION) && mydata.MDmic.containsLabel(EMDL_CTF_DETECTOR_PIXEL_SIZE))
		{
			FOR_ALL_OBJECTS_IN_METADATA_TABLE(mydata.MDmic)
			{
				mydata.MDimg.getValue(EMDL_CTF_MAGNIFICATION, mag);
				mydata.MDimg.getValue(EMDL_CTF_DETECTOR_PIXEL_SIZE, dstep);
				my_angpix = 10000. * dstep / mag;
				if (!has_magn)
				{
					first_angpix = my_angpix;
					has_magn = true;
				}
				else if (ABS(first_angpix - my_angpix) > 0.01)
				{
					REPORT_ERROR("MlOptimiser::initialiseGeneral: ERROR inconsistent magnification and detector pixel sizes in micrographs in input STAR file");
				}
			}
		}
		if (has_magn && ABS(first_angpix - mymodel.pixel_size) > 0.01)
		{
			if (verb > 0)
			{
				std::cout << "MlOptimiser::initialiseGeneral: WARNING modifying pixel size from " << mymodel.pixel_size << " to " << first_angpix << " based on magnification information in the input STAR file" << std::endl;
			}
			mymodel.pixel_size = first_angpix;
		}

	}
	// Expand movies if fn_data_movie is given AND we were not doing expanded movies already
	else if (fn_data_movie != "" && !do_realign_movies)
	{

		if (verb > 0)
		{
			std::cout << " Expanding current model for movie frames... " << std::endl;
		}

		do_realign_movies = true;
		nr_iter_wo_resol_gain = -1;
		nr_iter_wo_large_hidden_variable_changes = 0;
		smallest_changes_optimal_offsets = 999.;
		smallest_changes_optimal_orientations = 999.;
		current_changes_optimal_orientations = 999.;
		current_changes_optimal_offsets = 999.;

		// If we're realigning movie frames, then now read in the metadata of the movie frames and combine with the metadata of the average images
		mydata.expandToMovieFrames(fn_data_movie);

		// Now also modify the model to contain many more groups....
		// each groups has to become Nframes groups (get Nframes from new mydata)
		mymodel.expandToMovieFrames(mydata, movie_frame_running_avg_side);

		// Don't do norm correction for realignment of movies.
		do_norm_correction = false;

	}

	if (mymodel.nr_classes > 1 && do_split_random_halves)
	{
		REPORT_ERROR("ERROR: One cannot use --split_random_halves with more than 1 reference... You could first classify, and then refine each class separately using --random_halves.");
	}

	if (do_join_random_halves && !do_split_random_halves)
	{
		REPORT_ERROR("ERROR: cannot join random halves because they were not split in the previous run");
	}

	if (do_always_join_random_halves)
	{
		std::cout << " Joining half-reconstructions at each iteration: this is a developmental option to test sub-optimal FSC usage only! " << std::endl;
	}

	// If fn_tau is provided, read in the tau spectrum
	fix_tau = false;
	if (fn_tau != "None")
	{
		fix_tau = true;
		mymodel.readTauSpectrum(fn_tau, verb);
	}


	// Initialise the sampling object (sets prior mode and fills translations and rotations inside sampling object)
	sampling.initialise(mymodel.orientational_prior_mode, mymodel.ref_dim, false);

	// Default max_coarse_size is original size
	if (max_coarse_size < 0)
	{
		max_coarse_size = mymodel.ori_size;
	}

	if (particle_diameter < 0.)
	{
		particle_diameter = (mymodel.ori_size - width_mask_edge) * mymodel.pixel_size;
	}

	if (do_auto_refine)
	{
		nr_iter = 999;
		if(extra_iter!=0)
                        nr_iter = iter +extra_iter;
		has_fine_enough_angular_sampling = false;
	}

	// For do_average_unaligned, always use initial low_pass filter
	if (do_average_unaligned && ini_high < 0.)
	{
		// By default, use 0.07 dig.freq. low-pass filter
		// See S.H.W. Scheres (2010) Meth Enzym.
		ini_high = 1. / mymodel.getResolution(ROUND(0.07 * mymodel.ori_size));
	}

	// Fill tabulated sine and cosine tables
	tab_sin.initialise(5000);
	tab_cos.initialise(5000);

	// For skipped alignments: set nr_pool to one to have each thread work on one particle (with its own unique sampling arrays of 1 orientation and translation)
	// Also do not perturb this orientation, nor do oversampling or priors
	if (do_skip_align || do_skip_rotate)
	{
		mymodel.orientational_prior_mode = NOPRIOR;
		sampling.orientational_prior_mode = NOPRIOR;
		adaptive_oversampling = 0;
		nr_pool = max_nr_pool = 1;
		sampling.perturbation_factor = 0.;
		sampling.random_perturbation = 0.;
		sampling.setOneOrientation(0., 0., 0.);
		directions_have_changed = true;
		if (do_realign_movies)
		{
			nr_threads = 1;    // use only one thread, as there are no particles/orientations to parallelise anyway...
		}
		if (do_skip_align)
		{
			Matrix1D<DOUBLE> offset(2);
			sampling.setOneTranslation(offset);
		}
	}

	// Resize the pdf_direction arrays to the correct size and fill with an even distribution
	if (directions_have_changed)
	{
		mymodel.initialisePdfDirection(sampling.NrDirections(0, true));
	}

	// Initialise the wsum_model according to the mymodel
	wsum_model.initialise(mymodel, sampling.symmetryGroup());

	// Check that number of pooled particles is not larger than 1 for local angular searches
	// Because for local searches, each particle has a different set of nonzeroprior orientations, and thus a differently sized Mweight
	// If larger than 1, just reset to 1
	if (mymodel.orientational_prior_mode != NOPRIOR && max_nr_pool > 1)
	{
		if (verb > 0)
		{
			std::cout << " Performing local angular searches! Lowering max_nr_pool from " << max_nr_pool << " to 1!" << std::endl;
		}
		max_nr_pool = 1;
	}

	// Initialise sums of hidden variable changes
	// In later iterations, this will be done in updateOverallChangesInHiddenVariables
	sum_changes_optimal_orientations = 0.;
	sum_changes_optimal_offsets = 0.;
	sum_changes_optimal_classes = 0.;
	sum_changes_count = 0.;
	//gpt_image = NULL;

	// Skip scale correction if there are nor groups
	if (mymodel.nr_groups == 1)
	{
		do_scale_correction = false;
	}
	if (sub_extract)
	{
		if(fn_yellow_mask!=""){
			mymodel.do_yellow_red_mask = true;
			mymodel.yellow_mask.read(fn_yellow_mask);
		}
		if(fn_yellow_map!=""){
			mymodel.do_yellow_map_red_mask = true;
			mymodel.yellow_map.read(fn_yellow_map);
		}
		//mymodel.yellow_mask.read(fn_yellow_mask);
		mymodel.red_mask.read(fn_red_mask);
		//if(fn_yellow_map!="")
		//	mymodel.yellow_map.read(fn_yellow_map);
		mymodel.Iref_yellow.resize(mymodel.nr_classes);
		mymodel.Iref_red.resize(mymodel.nr_classes);
		
		for(int iclass = 0; iclass < mymodel.nr_classes; iclass++)
		{
			mymodel.Iref_yellow[iclass].resize(mymodel.Iref[iclass]);
			if(fn_yellow_map!="")
				mymodel.Iref_yellow.push_back(mymodel.yellow_map());
			mymodel.Iref_red[iclass].resize(mymodel.Iref[iclass]);
		}
	}
	if(fn_fsc_mask!="")
		mymodel.fsc_mask.read(fn_fsc_mask);
	// Check for rlnReconstructImageName in the data.star file. If it is present, set do_use_reconstruct_images to true
	do_use_reconstruct_images = mydata.MDimg.containsLabel(EMDL_IMAGE_RECONSTRUCT_NAME);
	if (do_use_reconstruct_images && verb > 0)
	{
		std::cout << " Using rlnReconstructImageName from the input data.star file!" << std::endl;
	}

#ifdef DEBUG
	std::cerr << "Leaving initialiseGeneral" << std::endl;
#endif

}

void MlOptimiser::initialiseWorkLoad()
{

	// Note, this function is overloaded in ml_optimiser_mpi...

	// Randomise the order of the particles
	if (random_seed == -1)
	{
		random_seed = time(NULL);
	}
	// This is for the division into random classes
	mydata.randomiseOriginalParticlesOrder(random_seed);
	// Also randomize random-number-generator for perturbations on the angles
	init_random_generator(random_seed);

	divide_equally(mydata.numberOfOriginalParticles(), 1, 0, my_first_ori_particle_id, my_last_ori_particle_id);

}

void MlOptimiser::calculateSumOfPowerSpectraAndAverageImage(MultidimArray<DOUBLE>& Mavg, bool myverb)
{

#ifdef DEBUG_INI
	std::cerr << "MlOptimiser::calculateSumOfPowerSpectraAndAverageImage Entering" << std::endl;
#endif

	int barstep, my_nr_ori_particles = my_last_ori_particle_id - my_first_ori_particle_id + 1;
	if (myverb > 0)
	{
		std::cout << " Estimating initial noise spectra " << std::endl;
		init_progress_bar(my_nr_ori_particles);
		barstep = XMIPP_MAX(1, my_nr_ori_particles / 60);
	}

	// Note the loop over the particles (part_id) is MPI-parallelized
	int nr_ori_particles_done = 0;
	Image<DOUBLE> img;
	FileName fn_img;
	MultidimArray<DOUBLE> ind_spectrum, sum_spectrum, count;
	// For spectrum calculation: recycle the transformer (so do not call getSpectrum all the time)
	MultidimArray<Complex > Faux;
	Matrix1D<DOUBLE> f(3);
	FourierTransformer transformer;
	MetaDataTable MDimg;

	for (long int ori_part_id = my_first_ori_particle_id; ori_part_id <= my_last_ori_particle_id; ori_part_id++, nr_ori_particles_done++)
	{

		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];

			for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
			{

				long int group_id = mydata.getGroupId(part_id, iseries);
				// TMP test for debuging
				if (group_id < 0 || group_id >= mymodel.nr_groups)
				{
					std::cerr << " group_id= " << group_id << std::endl;
					REPORT_ERROR("MlOptimiser::calculateSumOfPowerSpectraAndAverageImage: bad group_id");
				}

				// Extract the relevant MetaDataTable row from MDimg
				MDimg = mydata.getMetaDataImage(part_id, iseries);

				// Get the image filename
				MDimg.getValue(EMDL_IMAGE_NAME, fn_img);

				// Read image from disc
				img.read(fn_img);
				img().setXmippOrigin();

				// Check that the average in the noise area is approximately zero and the stddev is one
				if (!dont_raise_norm_error)
				{
					int bg_radius2 = ROUND(particle_diameter / (2. * mymodel.pixel_size));
					bg_radius2 *= bg_radius2;
					DOUBLE sum = 0.;
					DOUBLE sum2 = 0.;
					DOUBLE nn = 0.;
					FOR_ALL_ELEMENTS_IN_ARRAY3D(img())
					{
						if (k * k + i * i + j * j > bg_radius2)
						{
							sum += A3D_ELEM(img(), k, i, j);
							sum2 += A3D_ELEM(img(), k, i, j) * A3D_ELEM(img(), k, i, j);
							nn += 1.;
						}
					}
					// stddev
					sum2 -= sum * sum / nn;
					sum2 = sqrt(sum2 / nn);
					//average
					sum /= nn;

					// Average should be close to zero, i.e. max +/-50% of stddev...
					// Stddev should be close to one, i.e. larger than 0.5 and smaller than 2)
					if (ABS(sum / sum2) > 0.5 || sum2 < 0.5 || sum2 > 2.0)
					{
						std::cerr << " fn_img= " << fn_img << " bg_avg= " << sum << " bg_stddev= " << sum2 << std::endl;
						REPORT_ERROR("ERROR: It appears that these images have not been normalised to an average background value of 0 and a stddev value of 1. \n \
								Note that the average and stddev values for the background are calculated outside a circle with the particle diameter \n \
								You can use the relion_preprocess program to normalise your images \n \
								If you are sure you have normalised the images correctly (also see the RELION Wiki), you can switch off this error message using the --dont_check_norm command line option");
					}
				}

				// Apply a similar softMask as below (assume zero translations)
				if (do_zero_mask)
				{
					softMaskOutsideMap(img(), particle_diameter / (2. * mymodel.pixel_size), width_mask_edge);
				}

				// Calculate this image's power spectrum in: ind_spectrum
				ind_spectrum.initZeros(XSIZE(img()));
				count.initZeros(XSIZE(img()));
				// recycle the same transformer for all images
				transformer.FourierTransform(img(), Faux, false);
				FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Faux)
				{
					long int idx = ROUND(sqrt(kp * kp + ip * ip + jp * jp));
					ind_spectrum(idx) += norm(dAkij(Faux, k, i, j));
					count(idx) += 1.;
				}
				ind_spectrum /= count;

				// Resize the power_class spectrum to the correct size and keep sum
				ind_spectrum.resize(wsum_model.sigma2_noise[0]); // Store sum of all groups in group 0
				wsum_model.sigma2_noise[0] += ind_spectrum;
				wsum_model.sumw_group[0] += 1.;
				mymodel.nr_particles_group[group_id] += 1;


				// Also calculate average image
				if (part_id == mydata.ori_particles[my_first_ori_particle_id].particles_id[0])
				{
					Mavg = img();
				}
				else
				{
					Mavg += img();
				}

			} // end loop iseries
		} // end loop part_id (i)

		if (myverb > 0 && nr_ori_particles_done % barstep == 0)
		{
			progress_bar(nr_ori_particles_done);
		}

	} // end loop ori_part_id


	// Clean up the fftw object completely
	// This is something that needs to be done manually, as among multiple threads only one of them may actually do this
	transformer.cleanup();

	if (myverb > 0)
	{
		progress_bar(my_nr_ori_particles);
	}

#ifdef DEBUG_INI
	std::cerr << "MlOptimiser::calculateSumOfPowerSpectraAndAverageImage Leaving" << std::endl;
#endif

}

void MlOptimiser::setSigmaNoiseEstimatesAndSetAverageImage(MultidimArray<DOUBLE>& Mavg)
{

#ifdef DEBUG_INI
	std::cerr << "MlOptimiser::setSigmaNoiseEstimatesAndSetAverageImage Entering" << std::endl;
#endif

	// First calculate average image
	Mavg /= wsum_model.sumw_group[0];

	// for 2D refinements set 2D average to all references
	if (do_average_unaligned)
	{
		for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
		{
			mymodel.Iref[iclass] = Mavg;
		}
	}

	// Calculate sigma2_noise estimates as average of power class spectra, and subtract power spectrum of the average image from that
	if (do_calculate_initial_sigma_noise)
	{
		// Factor 2 because of 2-dimensionality of the complex plane
		mymodel.sigma2_noise[0] = wsum_model.sigma2_noise[0] / (2. * wsum_model.sumw_group[0]);

		// Calculate power spectrum of the average image
		MultidimArray<DOUBLE> spect;
		getSpectrum(Mavg, spect, POWER_SPECTRUM);
		spect /= 2.; // because of 2-dimensionality of the complex plane

		// Now subtract power spectrum of the average image from the average power spectrum of the individual images
		spect.resize(mymodel.sigma2_noise[0]);
		mymodel.sigma2_noise[0] -= spect;

		// Set the same spectrum for all groups
		for (int igroup = 0; igroup < mymodel.nr_groups; igroup++)
		{
			mymodel.sigma2_noise[igroup] = mymodel.sigma2_noise[0];
		}
	}


#ifdef DEBUG_INI
	std::cerr << "MlOptimiser::setSigmaNoiseEstimatesAndSetAverageImage Leaving" << std::endl;
#endif

}

void MlOptimiser::initialLowPassFilterReferences()
{
	if (ini_high > 0.)
	{

		// Make a soft (raised cosine) filter in Fourier space to prevent artefacts in real-space
		// The raised cosine goes through 0.5 at the filter frequency and has a width of width_mask_edge fourier pixels
		DOUBLE radius = mymodel.ori_size * mymodel.pixel_size / ini_high;
		radius -= WIDTH_FMASK_EDGE / 2.;
		DOUBLE radius_p = radius + WIDTH_FMASK_EDGE;
		FourierTransformer transformer;
		MultidimArray<Complex > Faux;
		for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
		{
			transformer.FourierTransform(mymodel.Iref[iclass], Faux);
			FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Faux)
			{
				DOUBLE r = sqrt((DOUBLE)(kp * kp + ip * ip + jp * jp));
				if (r < radius)
				{
					continue;
				}
				else if (r > radius_p)
				{
					DIRECT_A3D_ELEM(Faux, k, i, j) = 0.;
				}
				else
				{
					DIRECT_A3D_ELEM(Faux, k, i, j) *= 0.5 - 0.5 * cos(PI * (radius_p - r) / WIDTH_FMASK_EDGE);
				}
			}
			transformer.inverseFourierTransform(Faux, mymodel.Iref[iclass]);
		}

	}

}

/** ========================== EM-Iteration  ================================= */

void MlOptimiser::iterateSetup()
{

	// Make a barrier where all working threads wait
	global_barrier = new Barrier(nr_threads - 1);

	// Create threads to start working
	global_ThreadManager = new ThreadManager(nr_threads, this);

	// Set up the thread task distributors for the particles and the orientations (will be resized later on)
	exp_ipart_ThreadTaskDistributor = new ThreadTaskDistributor(1, 1);
	exp_iorient_ThreadTaskDistributor = new ThreadTaskDistributor(1, 1);

}
void MlOptimiser::iterateWrapUp()
{

	// delete barrier, threads and task distributors
	delete global_barrier;
	delete global_ThreadManager;
	delete exp_iorient_ThreadTaskDistributor;
	delete exp_ipart_ThreadTaskDistributor;

}

void MlOptimiser::iterate()
{

	if (do_split_random_halves)
	{
		REPORT_ERROR("ERROR: Cannot split data into random halves without using MPI!");
	}


	// launch threads etc
	iterateSetup();

	// Update the current resolution and image sizes, and precalculate resolution pointers
	// The rest of the time this will be done after maximization and before writing output files,
	// so that current resolution is in the output files of the current iteration
	updateCurrentResolution();

	bool has_already_reached_convergence = false;
	bool first = true;
	for (iter = iter + 1; iter <= nr_iter; iter++)
	{

#ifdef TIMING
		timer.tic(TIMING_EXP);
#endif

		// SA-stuff
		if (do_sim_anneal)
		{
			DOUBLE tau = -nr_iter / (std::log(temp_fin / temp_ini));
			temperature = temp_ini * exp(-iter / tau);
			std::cout << " temperature= " << temperature << std::endl;
		}

		if (do_auto_refine)
		{
			printConvergenceStats();
		}
		if (mode == 0)
		{
			expectation();
		}
		else if (mode == 1)
		{
			expectation_gpu();
		}
		else if (mode == 2)
		{
			expectation_gpu();
		}
		else
		{
			std::cout << " Error computing mode " << mode << std::endl;
		}




#ifdef TIMING
		timer.toc(TIMING_EXP);
		timer.tic(TIMING_MAX);
#endif

		if (do_skip_maximization)
		{
			// Only write data.star file and break from the iteration loop
			write(DONT_WRITE_SAMPLING, DO_WRITE_DATA, DONT_WRITE_OPTIMISER, DONT_WRITE_MODEL, 0);
			break;
		}

		if (mode == 0)
		{
			maximization();
		}
		else if (mode == 1)
		{
			maximization_gpu();
		}
		else if (mode == 2)
		{
			maximization_gpu();
		}
		else
		{
			std::cout << " Error computing mode " << mode << std::endl;
		}



#ifdef TIMING
		timer.toc(TIMING_MAX);
#endif

		// Apply masks to the reference images
		// At the last iteration, do not mask the map for validation purposes
		if (do_solvent && !has_converged)
		{
			solventFlatten();
		}

		// Re-calculate the current resolution, do this before writing to get the correct values in the output files
		updateCurrentResolution();

		// Write output files
		write(DO_WRITE_SAMPLING, DO_WRITE_DATA, DO_WRITE_OPTIMISER, DO_WRITE_MODEL, 0);

		if (do_auto_refine && has_converged)
		{
			if (verb > 0)
			{
				std::cout << " Auto-refine: Refinement has converged, stopping now... " << std::endl;
				std::cout << " Auto-refine: + Final reconstruction from all particles is saved as: " <<  fn_out << "_class001.mrc" << std::endl;
				std::cout << " Auto-refine: + Final model parameters are stored in: " << fn_out << "_model.star" << std::endl;
				std::cout << " Auto-refine: + Final data parameters are stored in: " << fn_out << "_data.star" << std::endl;
				std::cout << " Auto-refine: + Final resolution (without masking) is: " << 1. / mymodel.current_resolution << std::endl;
				if (acc_rot < 10.)
				{
					std::cout << " Auto-refine: + But you may want to run relion_postprocess to mask the unfil.mrc maps and calculate a higher resolution FSC" << std::endl;
				}
				else
				{
					std::cout << " Auto-refine: + WARNING: The angular accuracy is worse than 10 degrees, so basically you cannot align your particles!" << std::endl;
					std::cout << " Auto-refine: + WARNING: This has been observed to lead to spurious FSC curves, so be VERY wary of inflated resolution estimates..." << std::endl;
					std::cout << " Auto-refine: + WARNING: You most probably do NOT want to publish these results!" << std::endl;
					std::cout << " Auto-refine: + WARNING: Sometimes it is better to tune resolution yourself by adjusting T in a 3D-classification with a single class." << std::endl;
				}
			}
			break;
		}

		// Check whether we have converged by now
		// If we have, set do_join_random_halves and do_use_all_data for the next iteration
		if (do_auto_refine)
		{
			checkConvergence();
		}

#ifdef TIMING
		if (verb > 0)
		{
			timer.printTimes(false);
		}
#endif

	}

	// delete threads etc
	iterateWrapUp();
}

void MlOptimiser::expectation()
{

	//#define DEBUG_EXP
#ifdef DEBUG_EXP
	std::cerr << "Entering expectation" << std::endl;
#endif

	// Initialise some stuff
	// A. Update current size (may have been changed to ori_size in autoAdjustAngularSampling) and resolution pointers
	updateImageSizeAndResolutionPointers();

	// B. Initialise Fouriertransform, set weights in wsum_model to zero, etc
	expectationSetup();

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
		//std::cout << " mymodel.ref_dim: " << mymodel.ref_dim  << " n_trials_acc: " << n_trials_acc << std::endl;
		//REPORT_ERROR("Test break point!");
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
	while (nr_ori_particles_done < mydata.numberOfOriginalParticles())
	{


		my_first_ori_particle = nr_ori_particles_done;
		my_last_ori_particle = XMIPP_MIN(mydata.numberOfOriginalParticles() - 1, my_first_ori_particle + nr_pool - 1);

		// Get the metadata for these particles
		getMetaAndImageDataSubset(my_first_ori_particle, my_last_ori_particle); //Not parallelized, will be extracted before the loop
		// perform the actual expectation step on several particles
		expectationSomeParticles(my_first_ori_particle, my_last_ori_particle); // The parallel function
		// Set the metadata for these particles
		setMetaDataSubset(my_first_ori_particle, my_last_ori_particle);
		// Also monitor the changes in the optimal orientations and classes
		monitorHiddenVariableChanges(my_first_ori_particle, my_last_ori_particle);

		// Get the metadata for these particles

		nr_ori_particles_done += my_last_ori_particle - my_first_ori_particle + 1;

		if (verb > 0 && nr_ori_particles_done - prev_barstep > barstep)
		{
			prev_barstep = nr_ori_particles_done;
			progress_bar(nr_ori_particles_done);
		}
	}

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


void MlOptimiser::expectationSetup()
{
#ifdef DEBUG
	std::cerr << "Entering expectationSetup" << std::endl;
#endif

	// Re-initialise the random seed, because with a noisy_mask, inside the previous iteration different timings of different MPI nodes may have given rise to different number of calls to ran1
	// Use the iteration number so that each iteration has a different random seed
	init_random_generator(random_seed + iter);

	// Reset the random perturbation for this sampling
	sampling.resetRandomlyPerturbedSampling();

	// Initialise Projectors and fill vector with power_spectra for all classes
	mymodel.setFourierTransformMaps(!fix_tau, nr_threads);

	// Initialise all weighted sums to zero
	wsum_model.initZeros();

	project_backproject_GPU_data_init();


}

void MlOptimiser::expectationSetup_gpu()
{
#ifdef DEBUG
        std::cerr << "Entering expectationSetup" << std::endl;
#endif

        // Re-initialise the random seed, because with a noisy_mask, inside the previous iteration different timings of different MPI nodes may have given rise to different number of calls to ran1
        // Use the iteration number so that each iteration has a different random seed
        init_random_generator(random_seed + iter);

        // Reset the random perturbation for this sampling
        sampling.resetRandomlyPerturbedSampling();

        // Initialise Projectors and fill vector with power_spectra for all classes
        mymodel.setFourierTransformMaps_gpu(!fix_tau, nr_threads );
	 //sub_extract = false;
	 //std::cout << "init the PPref_red and yellow  0 " << std::endl;
        // Initialise all weighted sums to zero
        wsum_model.initZeros();
	//std::cout << "init the PPref_red and yellow  1" << std::endl;
	//printf("init red_project = %p\n", red_project_data_D);
	//red_project_data_D = NULL;
	//yellow_project_data_D = NULL;
        project_backproject_GPU_data_init();
	// std::cout << "after the PPref_red and yellow " << std::endl;


}

void MlOptimiser::expectationSetupCheckMemory(bool myverb)
{

	if (mymodel.orientational_prior_mode != NOPRIOR)
	{
		// First select one random direction and psi-angle for selectOrientationsWithNonZeroPriorProbability
		// This is to get an idea how many non-zero probabilities there will be
		DOUBLE ran_rot, ran_tilt, ran_psi;
		int randir = (int)(rnd_unif() * sampling.NrDirections(0, true));
		int ranpsi = (int)(rnd_unif() * sampling.NrPsiSamplings(0, true));
		if (randir == sampling.NrDirections(0, true))
		{
			//TMP
			REPORT_ERROR("RANDIR WAS TOO BIG!!!!");
			randir--;
		}
		if (ranpsi == sampling.NrPsiSamplings(0, true))
		{
			//TMP
			REPORT_ERROR("RANPSI WAS TOO BIG!!!!");
			ranpsi--;
		}
		sampling.getDirection(randir, ran_rot, ran_tilt);
		sampling.getPsiAngle(ranpsi, ran_psi);
		// Calculate local searches for these angles
		sampling.selectOrientationsWithNonZeroPriorProbability(ran_rot, ran_tilt, ran_psi,
		                                                       sqrt(mymodel.sigma2_rot), sqrt(mymodel.sigma2_tilt), sqrt(mymodel.sigma2_psi));
	}

	// Check whether things will fit into memory
	// Each DOUBLE takes 8 bytes, and their are mymodel.nr_classes references, express in Gb
	DOUBLE Gb = sizeof(DOUBLE) / (1024. * 1024. * 1024.);
	// A. Calculate approximate size of the reference maps
	// Forward projector has complex data, backprojector has complex data and real weight
	DOUBLE mem_references = Gb * mymodel.nr_classes * (2 * MULTIDIM_SIZE((mymodel.PPref[0]).data) + 3 * MULTIDIM_SIZE((wsum_model.BPref[0]).data));
	// B. Calculate size of the exp_Mweight matrices with (YSIZE=nr_pool, XSIZE=mymodel.nr_classes * sampling.NrSamplingPoints(adaptive_oversampling)
	nr_pool = max_nr_pool;

	if (mydata.maxNumberOfImagesPerOriginalParticle() > 1)
	{
		// Make sure that all particles in the data set have the same number of images
		// with the same transformation matrices so that their exp_R_mic can be re-used for pooled particles
		// If there are some particles with different transformations, then just set nr_pool to one
		// TODO: optimize this for randomised particle order.....
		// Currently that will lead to pretty bad efficiency IF there are multiple different tilt angles....
		// Or perhaps just forget about pooling. If we're re-refining the orientations that will be screwed anyway...

		// First find a particle with the maxNumberOfImagesPerParticle
		long int ref_part;
		long int maxn = mydata.maxNumberOfImagesPerOriginalParticle();
		for (ref_part = 0; ref_part < mydata.numberOfParticles(); ref_part++)
		{
			if (mydata.getNrImagesInSeries(ref_part) == maxn)
			{
				break;
			}
		}

		// Then check the transformation matrices for all the other particles are all the same
		// Note that particles are allowed to have fewer images in their series...
		Matrix2D<DOUBLE> first_R_mic, test_R_mic;
		bool is_ok = true;
		for (long int ipart = 0; ipart < mydata.numberOfParticles(); ipart++)
		{
			for (int iseries = 0; iseries < mydata.getNrImagesInSeries(ipart); iseries++)
			{
				first_R_mic = mydata.getMicrographTransformationMatrix(ref_part, iseries);
				test_R_mic = mydata.getMicrographTransformationMatrix(ipart, iseries);
				if (!first_R_mic.equal(test_R_mic))
				{
					is_ok = false;
					break;
				}
			}
		}
		if (!is_ok)
		{
			// Don't pool particles to prevent trouble when re-using exp_R_mic...
			nr_pool = 1;
			if (myverb > 0)
			{
				std::cout << " Switching off the pooling of particles because there are some series with distinct transformation matrices present in the data... ";
			}
		}
	}

	DOUBLE mem_pool = Gb * nr_pool * mymodel.nr_classes * sampling.NrSamplingPoints(adaptive_oversampling, false);
	// Estimate the rest of the program at 0.1 Gb?
	DOUBLE mem_rest = 0.1;
	DOUBLE total_mem_Gb_exp = mem_references + mem_pool + mem_rest;
	// Each reconstruction has to store 1 extra complex array (Fconv) and 4 extra DOUBLE arrays (Fweight, Fnewweight. vol_out and Mconv in convoluteBlobRealSpace),
	// in adddition to the DOUBLE weight-array and the complex data-array of the BPref
	// That makes a total of 2*2 + 5 = 9 * a DOUBLE array of size BPref
	DOUBLE total_mem_Gb_max = Gb * 9 * MULTIDIM_SIZE((wsum_model.BPref[0]).data);

	bool exp_does_not_fit = false;
	if (total_mem_Gb_exp > available_memory * nr_threads_original)
	{
		DOUBLE mem_for_pool = (available_memory * nr_threads_original) - mem_rest - mem_references;
		int suggested_nr_pool = FLOOR(mem_for_pool / (Gb * mymodel.nr_classes * sampling.NrSamplingPoints(adaptive_oversampling, true)));
		if (suggested_nr_pool > 0)
		{
			if (myverb > 0)
			{
				std::cout << "Reducing nr_pool to " << suggested_nr_pool << " to still fit into memory" << std::endl;
			}
			nr_pool = suggested_nr_pool;
			mem_pool = Gb * nr_pool * mymodel.nr_classes * sampling.NrSamplingPoints(adaptive_oversampling, false);
			total_mem_Gb_exp = mem_references + mem_pool + mem_rest;
		}
		else
		{
			exp_does_not_fit = true;
		}
	}

	if (myverb > 0)
	{
		// Calculate number of sampled hidden variables:
		int nr_ang_steps = CEIL(PI * particle_diameter * mymodel.current_resolution);
		DOUBLE myresol_angstep = 360. / nr_ang_steps;
		std::cout << " CurrentResolution= " << 1. / mymodel.current_resolution << " Angstroms, which requires orientationSampling of at least " << myresol_angstep
		          << " degrees for a particle of diameter " << particle_diameter << " Angstroms" << std::endl;
		for (int oversampling = 0; oversampling <= adaptive_oversampling; oversampling++)
		{
			std::cout << " Oversampling= " << oversampling << " NrHiddenVariableSamplingPoints= " << mymodel.nr_classes* sampling.NrSamplingPoints(oversampling, true) << std::endl;
			std::cout << " OrientationalSampling= " << sampling.getAngularSampling(oversampling)
			          << " NrOrientations= " << sampling.NrDirections(oversampling, false)*sampling.NrPsiSamplings(oversampling, false) << std::endl;
			std::cout << " TranslationalSampling= " << sampling.getTranslationalSampling(oversampling)
			          << " NrTranslations= " << sampling.NrTranslationalSamplings(oversampling) << std::endl;
			std::cout << "=============================" << std::endl;
		}
	}

	if (myverb > 0)
	{
		std::cout << " Estimated memory for expectation step  > " << total_mem_Gb_exp << " Gb, available memory = " << available_memory* nr_threads_original << " Gb." << std::endl;
		std::cout << " Estimated memory for maximization step > " << total_mem_Gb_max << " Gb, available memory = " << available_memory* nr_threads_original << " Gb." << std::endl;

		if (total_mem_Gb_max > available_memory * nr_threads_original || exp_does_not_fit)
		{
			if (exp_does_not_fit)
			{
				std::cout << " WARNING!!! Expected to run out of memory during expectation step ...." << std::endl;
			}
			if (total_mem_Gb_max > available_memory * nr_threads_original)
			{
				std::cout << " WARNING!!! Expected to run out of memory during maximization step ...." << std::endl;
			}
			std::cout << " WARNING!!! Did you set --memory_per_thread to reflect the number of Gb per core on your computer?" << std::endl;
			std::cout << " WARNING!!! If so, then check your processes are not swapping and consider running fewer MPI processors per node." << std::endl;
			std::cout << " + Available memory for each thread, as given by --memory_per_thread      : " << available_memory << " Gb" << std::endl;
			std::cout << " + Number of threads used per MPI process, as given by --j                : " << nr_threads_original << std::endl;
			std::cout << " + Available memory per MPI process 										: " << available_memory* nr_threads_original << " Gb" << std::endl;
		}
	}

	// Now that we also have nr_pool, resize the task manager for the particles

	/// When there are multiple particles for each ori_particle, then this ThreadTaskDistributor will again be resized somewhere below
	exp_ipart_ThreadTaskDistributor->resize(nr_pool, 1);

	// Also resize task manager for the orientations in case of NOPRIOR (otherwise resizing is done in doThreadGetFourierTransformsAndCtfs)
	if (do_skip_align || do_skip_rotate)
	{
		exp_iorient_ThreadTaskDistributor->resize(1, 1);
	}
	else if (mymodel.orientational_prior_mode == NOPRIOR)
	{
		long int nr_orients = sampling.NrDirections() * sampling.NrPsiSamplings();
		int threadBlockSize = (nr_orients > 100) ? 10 : 1;
		exp_iorient_ThreadTaskDistributor->resize(nr_orients, threadBlockSize);
	}
#ifdef DEBUG
	std::cerr << "Leaving expectationSetup" << std::endl;
#endif

}

void MlOptimiser::expectationSomeParticles(long int my_first_ori_particle, long int my_last_ori_particle)
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


	//          std::cout << "iclass_min = " << iclass_min << " iclass_max = " << iclass_max << std::endl;
	//          std::cout << nr_pool << std::endl;
	//          std::cout << mydata.numberOfOriginalParticles() << " " << mymodel.nr_classes << " " << exp_my_first_ori_particle << std::endl;
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

	exp_ipart_ThreadTaskDistributor->reset();
	global_ThreadManager->run(globalGetFourierTransformsAndCtfs);

	//doThreadGetFourierTransformsAndCtfs
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
		getAllSquaredDifferences();


#ifdef DEBUG_ESP_MEM
		std::cerr << "After getAllSquaredDifferences, use top to see memory usage and then press any key to continue... " << std::endl;
		std::cin >> c;
#endif
		// Now convert the squared difference terms to weights,
		// also calculate exp_sum_weight, and in case of adaptive oversampling also exp_significant_weight
		convertAllSquaredDifferencesToWeights();


#ifdef DEBUG_ESP_MEM
		std::cerr << "After convertAllSquaredDifferencesToWeights, press any key to continue... " << std::endl;
		std::cin >> c;
#endif

	}// end loop over 2 exp_ipass iterations


	// For the reconstruction step use mymodel.current_size!
	exp_current_image_size = mymodel.current_size;

#ifdef DEBUG_ESP_MEM
	std::cerr << "Before storeWeightedSums, press any key to continue... " << std::endl;
	std::cin >> c;
#endif
	storeWeightedSums();

	// Now calculate the optimal translation for each of the individual images in the series
	//if (mydata.maxNumberOfImagesPerOriginalParticle(my_first_ori_particle, my_last_ori_particle) > 1 && !(do_firstiter_cc && iter == 1))
	//  getOptimalOrientationsForIndividualImagesInSeries();

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

void MlOptimiser::maximization()
{

	if (verb > 0)
	{
		std::cout << " Maximization ..." << std::endl;
		init_progress_bar(mymodel.nr_classes);
	}

	// First reconstruct the images for each class
	for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
	{
		if (mymodel.pdf_class[iclass] > 0.)
		{
			(wsum_model.BPref[iclass]).reconstruct(mymodel.Iref[iclass], gridding_nr_iter, do_map,
			                                       mymodel.tau2_fudge_factor, mymodel.tau2_class[iclass], mymodel.sigma2_class[iclass],
			                                       mymodel.data_vs_prior_class[iclass], mymodel.fsc_halves_class[iclass], wsum_model.pdf_class[iclass],
			                                       false, false, nr_threads, minres_map);

		}
		else
		{
			mymodel.Iref[iclass].initZeros();
		}

		if (verb > 0)
		{
			progress_bar(iclass);
		}
	}

	// Then perform the update of all other model parameters
	maximizationOtherParameters();

	// Keep track of changes in hidden variables
	updateOverallChangesInHiddenVariables();

	// This doesn't really work, and I need the original priors for the polishing...
	//if (do_realign_movies)
	//  updatePriorsForMovieFrames();

	if (verb > 0)
	{
		progress_bar(mymodel.nr_classes);
	}

}

void MlOptimiser::maximizationOtherParameters()
{
	// Note that reconstructions are done elsewhere!
#ifdef DEBUG
	std::cerr << "Entering maximizationOtherParameters" << std::endl;
#endif

	// Calculate total sum of weights, and average CTF for each class (for SSNR estimation)
	DOUBLE sum_weight = 0.;
	for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
	{
		sum_weight += wsum_model.pdf_class[iclass];
	}

	// Update average norm_correction
	if (do_norm_correction)
	{
		mymodel.avg_norm_correction = wsum_model.avg_norm_correction / sum_weight;
	}

	if (do_scale_correction && !(iter == 1 && do_firstiter_cc))
	{
		DOUBLE avg_scale_correction = 0., nr_part = 0.;
		for (int igroup = 0; igroup < mymodel.nr_groups; igroup++)
		{

#ifdef DEVEL_BFAC
			// TMP
			if (verb > 0)
			{
				for (int i = 0; i < XSIZE(wsum_model.wsum_signal_product_spectra[igroup]); i++)
				{
					std::cout << " igroup= " << igroup << " i= " << i << " " << wsum_model.wsum_signal_product_spectra[igroup](i) << " " << wsum_model.wsum_reference_power_spectra[igroup](i) << std::endl;
				}
			}
#endif

			DOUBLE sumXA = wsum_model.wsum_signal_product_spectra[igroup].sum();
			DOUBLE sumAA = wsum_model.wsum_reference_power_spectra[igroup].sum();
			if (sumAA > 0.)
			{
				mymodel.scale_correction[igroup] = sumXA / sumAA;
			}
			else
			{
				mymodel.scale_correction[igroup] = 1.;
			}
			avg_scale_correction += (DOUBLE)(mymodel.nr_particles_group[igroup]) * mymodel.scale_correction[igroup];
			nr_part += (DOUBLE)(mymodel.nr_particles_group[igroup]);

		}

		// Constrain average scale_correction to one.
		avg_scale_correction /= nr_part;
		for (int igroup = 0; igroup < mymodel.nr_groups; igroup++)
		{
			mymodel.scale_correction[igroup] /= avg_scale_correction;
			//#define DEBUG_UPDATE_SCALE
#ifdef DEBUG_UPDATE_SCALE
			if (verb > 0)
			{
				std::cerr << "Group " << igroup + 1 << ": scale_correction= " << mymodel.scale_correction[igroup] << std::endl;
				for (int i = 0; i < XSIZE(wsum_model.wsum_reference_power_spectra[igroup]); i++)
					if (wsum_model.wsum_reference_power_spectra[igroup](i) > 0.)
						std::cerr << " i= " << i << " XA= " << wsum_model.wsum_signal_product_spectra[igroup](i)
						          << " A2= " << wsum_model.wsum_reference_power_spectra[igroup](i)
						          << " XA/A2= " << wsum_model.wsum_signal_product_spectra[igroup](i) / wsum_model.wsum_reference_power_spectra[igroup](i) << std::endl;

			}
#endif
		}

	}

	// Update model.pdf_class vector (for each k)
	for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
	{
		mymodel.pdf_class[iclass] = wsum_model.pdf_class[iclass] / sum_weight;

		// for 2D also update priors of translations for each class!
		if (mymodel.ref_dim == 2)
		{
			if (wsum_model.pdf_class[iclass] > 0.)
			{
				mymodel.prior_offset_class[iclass] = wsum_model.prior_offset_class[iclass] / wsum_model.pdf_class[iclass];
			}
			else
			{
				mymodel.prior_offset_class[iclass].initZeros();
			}
		}

		// Use sampling.NrDirections(0, true) to include all directions (also those with zero prior probability for any given image)
		for (int idir = 0; idir < sampling.NrDirections(0, true); idir++)
		{
			mymodel.pdf_direction[iclass](idir) = wsum_model.pdf_direction[iclass](idir) / sum_weight;
		}
	}

	// Update sigma2_offset
	// Factor 2 because of the 2-dimensionality of the xy-plane
	if (!fix_sigma_offset)
	{
		mymodel.sigma2_offset = (wsum_model.sigma2_offset) / (2. * sum_weight);
	}

	// TODO: update estimates for sigma2_rot, sigma2_tilt and sigma2_psi!

	// Also refrain from updating sigma_noise after the first iteration with first_iter_cc!
	if (!fix_sigma_noise && !(iter == 1 && do_firstiter_cc))
	{
		for (int igroup = 0; igroup < mymodel.nr_groups; igroup++)
		{
			// Factor 2 because of the 2-dimensionality of the complex-plane
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mymodel.sigma2_noise[igroup])
			{
				DIRECT_MULTIDIM_ELEM(mymodel.sigma2_noise[igroup], n) =
				    DIRECT_MULTIDIM_ELEM(wsum_model.sigma2_noise[igroup], n) /
				    (2. * wsum_model.sumw_group[igroup] * DIRECT_MULTIDIM_ELEM(Npix_per_shell, n));
			}
		}
	}

	// After the first iteration the references are always CTF-corrected
	if (do_ctf_correction)
	{
		refs_are_ctf_corrected = true;
	}

	// Some statistics to output
	mymodel.LL =    wsum_model.LL;
	if ((iter == 1 && do_firstiter_cc) || do_always_cc)
	{
		mymodel.LL /= sum_weight;    // this now stores the average ccf
	}
	mymodel.ave_Pmax = wsum_model.ave_Pmax / sum_weight;

	// After the first, special iteration, apply low-pass filter of -ini_high again
	if (iter == 1 && do_firstiter_cc)
	{
		initialLowPassFilterReferences();
		if (ini_high > 0.)
		{
			// Adjust the tau2_class and data_vs_prior_class, because they were calculated on the unfiltered maps
			// This is merely a matter of having correct output in the model.star file (these values are not used in the calculations)
			DOUBLE radius = mymodel.ori_size * mymodel.pixel_size / ini_high;
			radius -= WIDTH_FMASK_EDGE / 2.;
			DOUBLE radius_p = radius + WIDTH_FMASK_EDGE;

			for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
			{
				for (int rr = 0; rr < XSIZE(mymodel.tau2_class[iclass]); rr++)
				{
					DOUBLE r = (DOUBLE)rr;
					if (r < radius)
					{
						continue;
					}
					else if (r > radius_p)
					{
						DIRECT_A1D_ELEM(mymodel.tau2_class[iclass], rr) = 0.;
						DIRECT_A1D_ELEM(mymodel.data_vs_prior_class[iclass], rr) = 0.;
					}
					else
					{
						DOUBLE raisedcos = 0.5 - 0.5 * cos(PI * (radius_p - r) / WIDTH_FMASK_EDGE);
						DIRECT_A1D_ELEM(mymodel.tau2_class[iclass], rr) *= raisedcos * raisedcos;
						DIRECT_A1D_ELEM(mymodel.data_vs_prior_class[iclass], rr) *= raisedcos * raisedcos;
					}
				}
			}
		}

		if (do_generate_seeds && mymodel.nr_classes > 1)
		{
			// In the first CC-iteration only a single reference was used
			// Now copy this one reference to all K references, for seed generation in the second iteration
			for (int iclass = 1; iclass < mymodel.nr_classes; iclass++)
			{
				mymodel.tau2_class[iclass] =  mymodel.tau2_class[0];
				mymodel.data_vs_prior_class[iclass] = mymodel.data_vs_prior_class[0];
				mymodel.pdf_class[iclass] = mymodel.pdf_class[0] / mymodel.nr_classes;
				mymodel.pdf_direction[iclass] = mymodel.pdf_direction[0];
				mymodel.Iref[iclass] = mymodel.Iref[0];
			}
			mymodel.pdf_class[0] /= mymodel.nr_classes;
		}

	}

#ifdef DEBUG
	std::cerr << "Leaving maximizationOtherParameters" << std::endl;
#endif
}


void MlOptimiser::solventFlatten()
{
#ifdef DEBUG
	std::cerr << "Entering MlOptimiser::solventFlatten" << std::endl;
#endif
	// First read solvent mask from disc, or pre-calculate it
	Image<DOUBLE> Isolvent, Isolvent2;
	Isolvent().resize(mymodel.Iref[0]);
	Isolvent().setXmippOrigin();
	Isolvent().initZeros();
	if (fn_mask.contains("None"))
	{
		DOUBLE radius = particle_diameter / (2. * mymodel.pixel_size);
		DOUBLE radius_p = radius + width_mask_edge;
		FOR_ALL_ELEMENTS_IN_ARRAY3D(Isolvent())
		{
			DOUBLE r = sqrt((DOUBLE)(k * k + i * i + j * j));
			if (r < radius)
			{
				A3D_ELEM(Isolvent(), k, i, j) = 1.;
			}
			else if (r > radius_p)
			{
				A3D_ELEM(Isolvent(), k, i, j) = 0.;
			}
			else
			{
				A3D_ELEM(Isolvent(), k, i, j) = 0.5 - 0.5 * cos(PI * (radius_p - r) / width_mask_edge);
			}
		}
	}
	else
	{
		Isolvent.read(fn_mask);
		Isolvent().setXmippOrigin();

		if (Isolvent().computeMin() < 0. || Isolvent().computeMax() > 1.)
		{
			REPORT_ERROR("MlOptimiser::solventFlatten: ERROR solvent mask should contain values between 0 and 1 only...");
		}
	}

	// Also read a second solvent mask if necessary
	if (!fn_mask2.contains("None"))
	{
		Isolvent2.read(fn_mask2);
		Isolvent2().setXmippOrigin();
		if (!Isolvent2().sameShape(Isolvent()))
		{
			REPORT_ERROR("MlOptimiser::solventFlatten ERROR: second solvent mask is of incorrect size.");
		}
	}

	for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
	{

		// Then apply the expanded solvent mask to the map
		mymodel.Iref[iclass] *= Isolvent();

		// Apply a second solvent mask if necessary
		// This may for example be useful to set the interior of icosahedral viruses to a constant density value that is higher than the solvent
		// Invert the solvent mask, so that an input mask can be given where 1 is the masked area and 0 is protein....
		if (!fn_mask2.contains("None"))
		{
			softMaskOutsideMap(mymodel.Iref[iclass], Isolvent2(), true);
		}

	} // end for iclass
#ifdef DEBUG
	std::cerr << "Leaving MlOptimiser::solventFlatten" << std::endl;
#endif

}

void MlOptimiser::updateCurrentResolution()
{
	//#define DEBUG
#ifdef DEBUG
	std::cerr << "Entering MlOptimiser::updateCurrentResolution" << std::endl;
#endif


	int maxres = 0;
	if (do_map)
	{
		// Set current resolution
		if (ini_high > 0. && (iter == 0 || (iter == 1 && do_firstiter_cc)))
		{
			maxres = ROUND(mymodel.ori_size * mymodel.pixel_size / ini_high);
		}
		else
		{
			// Calculate at which resolution shell the data_vs_prior drops below 1
			int ires;
			for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
			{
				for (ires = 1; ires < mymodel.ori_size / 2; ires++)
				{
					if (DIRECT_A1D_ELEM(mymodel.data_vs_prior_class[iclass], ires) < 1.)
					{
						break;
					}
				}
				// Subtract one shell to be back on the safe side
				ires--;
				if (ires > maxres)
				{
					maxres = ires;
				}
			}

			// Never allow smaller maxres than minres_map
			maxres = XMIPP_MAX(maxres, minres_map);
		}
	}
	else
	{
		// If we are not doing MAP-estimation, set maxres to Nyquist
		maxres = mymodel.ori_size / 2;
	}
	DOUBLE newres = mymodel.getResolution(maxres);


	// Check whether resolution improved, if not increase nr_iter_wo_resol_gain
	//if (newres <= best_resol_thus_far)
	if (newres <= mymodel.current_resolution + 0.0001) // Add 0.0001 to avoid problems due to rounding error
	{
		nr_iter_wo_resol_gain++;
	}
	else
	{
		nr_iter_wo_resol_gain = 0;
	}

	// Store best resolution thus far (but no longer do anything with it anymore...)
	if (newres > best_resol_thus_far)
	{
		best_resol_thus_far = newres;
	}

	mymodel.current_resolution = newres;

}

void MlOptimiser::updateImageSizeAndResolutionPointers()
{

	// Increment the current_size
	// If we are far from convergence (in the initial stages of refinement) take steps of 25% the image size
	// Do this whenever the FSC at the current_size is larger than 0.2, but NOT when this is in combination with very low Pmax values,
	// in the latter case, over-marginalisation may lead to spuriously high FSCs (2 smoothed maps may look very similar at high-res: all zero!)
	//
	int maxres = mymodel.getPixelFromResolution(mymodel.current_resolution);
	if (mymodel.ave_Pmax > 0.1 && has_high_fsc_at_limit)
	{
		maxres += ROUND(0.25 * mymodel.ori_size / 2);
	}
	else
	{
		// If we are near our resolution limit, use incr_size (by default 10 shells)
		maxres += incr_size;
	}

	// Go back from resolution shells (i.e. radius) to image size, which are BTW always even...
	mymodel.current_size = maxres * 2;

	// If realigning movies: go all the way because resolution increase may be substantial
	if (do_use_all_data||(iter==nr_iter&&do_auto_refine))
	{
		mymodel.current_size = mymodel.ori_size;
	}

	// current_size can never be larger than ori_size:
	mymodel.current_size = XMIPP_MIN(mymodel.current_size, mymodel.ori_size);
	// The current size is also used in wsum_model (in unpacking)
	wsum_model.current_size = mymodel.current_size;

	// Update coarse_size
	if (strict_highres_exp > 0.)
	{
		// Strictly limit the coarse size to the one corresponding to strict_highres_exp
		coarse_size = 2 * ROUND(mymodel.ori_size * mymodel.pixel_size / strict_highres_exp);
	}
	else if (adaptive_oversampling > 0.)
	{
		// Dependency of coarse_size on the angular sampling used in the first pass
		DOUBLE rotated_distance = (sampling.getAngularSampling() / 360.) * PI * particle_diameter;
		DOUBLE keepsafe_factor = (mymodel.ref_dim == 3) ? 1.2 : 1.5;
		DOUBLE coarse_resolution = rotated_distance / keepsafe_factor;
		// Note coarse_size should be even-valued!
		coarse_size = 2 * CEIL(mymodel.pixel_size * mymodel.ori_size / coarse_resolution);
		// Coarse size can never be larger than max_coarse_size
		coarse_size = XMIPP_MIN(max_coarse_size, coarse_size);
	}
	else
	{
		coarse_size = mymodel.current_size;
	}
	// Coarse_size can never become bigger than current_size
	coarse_size = XMIPP_MIN(mymodel.current_size, coarse_size);

	/// Also update the resolution pointers here

	// Calculate number of pixels per resolution shell
	Npix_per_shell.initZeros(mymodel.ori_size / 2 + 1);
	MultidimArray<DOUBLE> aux;
	aux.resize(mymodel.ori_size, mymodel.ori_size / 2 + 1);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(aux)
	{
		int ires = ROUND(sqrt((DOUBLE)(kp * kp + ip * ip + jp * jp)));
		// TODO: better check for volume_refine, but the same still seems to hold... Half of the yz plane (either ip<0 or kp<0 is redundant at jp==0)
		// Exclude points beyond XSIZE(Npix_per_shell), and exclude half of the x=0 column that is stored twice in FFTW
		if (ires < mymodel.ori_size / 2 + 1 && !(jp == 0 && ip < 0))
		{
			Npix_per_shell(ires) += 1;
		}
	}

	Mresol_fine.resize(mymodel.current_size, mymodel.current_size / 2 + 1);
	Mresol_fine.initConstant(-1);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Mresol_fine)
	{
		int ires = ROUND(sqrt((DOUBLE)(kp * kp + ip * ip + jp * jp)));
		// TODO: better check for volume_refine, but the same still seems to hold... Half of the yz plane (either ip<0 or kp<0 is redundant at jp==0)
		// Exclude points beyond ires, and exclude and half (y<0) of the x=0 column that is stored twice in FFTW
		if (ires < mymodel.current_size / 2 + 1  && !(jp == 0 && ip < 0))
		{
			DIRECT_A3D_ELEM(Mresol_fine, k, i, j) = ires;
		}
	}

	Mresol_coarse.resize(coarse_size, coarse_size / 2 + 1);
	Mresol_coarse.initConstant(-1);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Mresol_coarse)
	{
		int ires = ROUND(sqrt((DOUBLE)(kp * kp + ip * ip + jp * jp)));
		// Exclude points beyond ires, and exclude and half (y<0) of the x=0 column that is stored twice in FFTW
		// exclude lowest-resolution points
		if (ires < coarse_size / 2 + 1 && !(jp == 0 && ip < 0))
		{
			DIRECT_A3D_ELEM(Mresol_coarse, k, i, j) = ires;
		}
	}

	//#define DEBUG_MRESOL
#ifdef DEBUG_MRESOL
	Image<DOUBLE> img;
	img().resize(YSIZE(Mresol_fine), XSIZE(Mresol_fine));
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img())
	{
		DIRECT_MULTIDIM_ELEM(img(), n) = (DOUBLE)DIRECT_MULTIDIM_ELEM(Mresol_fine, n);
	}
	img.write("Mresol_fine.mrc");
	img().resize(YSIZE(Mresol_coarse), XSIZE(Mresol_coarse));
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img())
	{
		DIRECT_MULTIDIM_ELEM(img(), n) = (DOUBLE)DIRECT_MULTIDIM_ELEM(Mresol_coarse, n);
	}
	img.write("Mresol_coarse.mrc");
#endif


#ifdef DEBUG
	std::cerr << " current_size= " << mymodel.current_size << " coarse_size= " << coarse_size << " current_resolution= " << mymodel.current_resolution << std::endl;
	std::cerr << "Leaving MlOptimiser::updateCurrentResolution" << std::endl;
#endif

}


DOUBLE MlOptimiser::calculatePdfOffset(Matrix1D<DOUBLE> offset, Matrix1D<DOUBLE> prior)
{
	if (mymodel.sigma2_offset < 0.0001)
	{
		return (offset.sum2() > 0.) ? 0. : 1.;
	}
	else
	{
		return exp((offset - prior).sum2() / (-2. * mymodel.sigma2_offset)) / (2. * PI * mymodel.sigma2_offset);
	}
}

void MlOptimiser::calculateRunningAveragesOfMovieFrames()
{
	std::vector<MultidimArray<Complex > > runavg_Fimgs;
	std::vector<int> count_runavg;
	MultidimArray<Complex > Fzero;
	Fzero.resize(exp_Fimgs[0]);
	Fzero.initZeros();

	// initialise the sums at zero
	for (int iimg = 0; iimg < exp_Fimgs.size(); iimg++)
	{
		runavg_Fimgs.push_back(Fzero);
		count_runavg.push_back(0);
	}

	// running avgs NOT for series!
	int iseries = 0;

	//#define DEBUG_RUNAVG
#ifdef DEBUG_RUNAVG
	FourierTransformer transformer;
	MultidimArray< Complex > Fimg;
	Image<DOUBLE> It;
	if (verb)
	{
		Fimg = exp_Fimgs[0];
		It().resize(YSIZE(Fimg), YSIZE(Fimg));
		transformer.inverseFourierTransform(Fimg, It());
		CenterFFT(It(), false);
		It.write("Fimg.spi");
		std::cerr << "Written Fimg" << std::endl;
	}
#endif

	// Calculate the running sums
	for (int iimg = 0; iimg < exp_Fimgs.size(); iimg++)
	{
		// Who are we?
		int my_ipart = exp_iimg_to_ipart[iimg];
		long int my_ori_part_id = exp_ipart_to_ori_part_id[my_ipart];
		long int my_part_id = exp_ipart_to_part_id[my_ipart];
		int my_frame = exp_ipart_to_ori_part_nframe[my_ipart];

#ifdef DEBUG_RUNAVG
		if (verb)
		{
			long int my_img_id = mydata.getImageId(my_part_id, iseries);
			FileName fntt;
			mydata.MDimg.getValue(EMDL_IMAGE_NAME, fntt, my_img_id);
			std::cerr << " my= " << fntt;
		}
#endif

		long int my_first_runavg_frame = XMIPP_MAX(0, my_frame - movie_frame_running_avg_side);
		long int my_last_runavg_frame = XMIPP_MIN(mydata.ori_particles[my_ori_part_id].particles_id.size() - 1, my_frame + movie_frame_running_avg_side);

		// Run over all images again and see which ones to sum
		for (int iimg2 = 0; iimg2 < exp_Fimgs.size(); iimg2++)
		{
			int other_ipart = exp_iimg_to_ipart[iimg2];
			long int other_ori_part_id = exp_ipart_to_ori_part_id[other_ipart];
			long int other_part_id = exp_ipart_to_part_id[other_ipart];
			int other_frame = exp_ipart_to_ori_part_nframe[other_ipart];

			if (my_ori_part_id == other_ori_part_id && other_frame >= my_first_runavg_frame && other_frame <= my_last_runavg_frame)
			{
#ifdef DEBUG_RUNAVG
				if (verb)
				{
					long int other_img_id = mydata.getImageId(other_part_id, iseries);
					FileName fnt, fnm, fnp;
					mydata.MDimg.getValue(EMDL_IMAGE_NAME, fnt, other_img_id);
					mydata.MDimg.getValue(EMDL_PARTICLE_ORI_NAME, fnp, other_img_id);
					mydata.MDimg.getValue(EMDL_MICROGRAPH_NAME, fnm, other_img_id);
					std::cerr << " = " << fnt << " " << fnm << " " << fnp;
				}
#endif

				// Add to sum
				runavg_Fimgs[iimg] += exp_Fimgs[iimg2];
				count_runavg[iimg] += 1;
			}
		}

#ifdef DEBUG_RUNAVG
		if (verb)
		{
			std::cerr << std::endl;
		}
#endif
	}

	// Calculate averages from sums and set back in exp_ vectors
	for (int iimg = 0; iimg < exp_Fimgs.size(); iimg++)
	{
		DOUBLE sum = (DOUBLE)count_runavg[iimg];
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(exp_Fimgs[iimg])
		{
			DIRECT_MULTIDIM_ELEM(exp_Fimgs[iimg], n) = DIRECT_MULTIDIM_ELEM(runavg_Fimgs[iimg], n) / sum;
		}
		// Also lower the power of the images for the sigma2_noise and diff2 calculations beyond current_size....
		// sigma2_(a+b) = sigma2_(a) + sigma2_(b)
		// The actual values are lost, just hope the images obey statistics...
		exp_power_imgs[iimg] /= sum;
		exp_highres_Xi2_imgs[iimg] /= sum;
	}
#ifdef DEBUG_RUNAVG
	if (verb)
	{
		Fimg = exp_Fimgs[0];
		It().resize(YSIZE(Fimg), YSIZE(Fimg));
		transformer.inverseFourierTransform(Fimg, It());
		CenterFFT(It(), false);
		It.write("Frunavg.spi");
		std::cerr << "Written Frunavg.spi, sleeping 2 seconds..." << std::endl;
		sleep(2);

	}
#endif

}

void MlOptimiser::doThreadGetFourierTransformsAndCtfs(int thread_id)
{
	// Only first thread initialises
	if (thread_id == 0)
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

		// Resize to the right size instead of using pushbacks
		exp_starting_image_no.resize(exp_nr_particles);

		// First check how many images there are in the series for each particle...
		// And calculate exp_nr_images
		exp_nr_images = 0;
		for (long int ori_part_id = exp_my_first_ori_particle, my_image_no = 0, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
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

	}
	global_barrier->wait();

	FourierTransformer transformer;
	size_t first_ipart = 0, last_ipart = 0;
	while (exp_ipart_ThreadTaskDistributor->getTasks(first_ipart, last_ipart))
	{

		for (long int ipart = first_ipart; ipart <= last_ipart; ipart++)
		{
			// the exp_ipart_ThreadTaskDistributor was set with nr_pool,
			// but some, e.g. the last, batch of pooled particles may be smaller
			if (ipart >= exp_nr_particles)
			{
				break;
			}

#ifdef DEBUG_CHECKSIZES
			if (ipart >= exp_ipart_to_part_id.size())
			{
				std::cerr << "ipart= " << ipart << " exp_ipart_to_part_id.size()= " << exp_ipart_to_part_id.size() << std::endl;
				REPORT_ERROR("ipart >= exp_ipart_to_part_id.size()");
			}
#endif
			long int part_id = exp_ipart_to_part_id[ipart];

			// Prevent movies and series at the same time...
			if (mydata.getNrImagesInSeries(part_id) > 1 && do_realign_movies)
			{
				REPORT_ERROR("Not ready yet for dealing with image series at the same time as realigning movie frames....");
			}

			for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
			{

				FileName fn_img;
				Image<DOUBLE> img, rec_img;
				MultidimArray<Complex > Fimg, Faux;
				MultidimArray<DOUBLE> Fctf;
				int my_image_no = exp_starting_image_no.at(ipart) + iseries;
				// Which group do I belong?
				int group_id = mydata.getGroupId(part_id, iseries);

				// Get the norm_correction
				DOUBLE normcorr = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM);

				// Get the optimal origin offsets from the previous iteration
				Matrix1D<DOUBLE> my_old_offset(2), my_prior(2);
				XX(my_old_offset) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF);
				YY(my_old_offset) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF);
				XX(my_prior)      = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF_PRIOR);
				YY(my_prior)      = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF_PRIOR);
				// Uninitialised priors were set to 999.
				if (XX(my_prior) > 998.99 && XX(my_prior) < 999.01)
				{
					XX(my_prior) = 0.;
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

					global_mutex.lock();

					// Select only those orientations that have non-zero prior probability
					sampling.selectOrientationsWithNonZeroPriorProbability(prior_rot, prior_tilt, prior_psi,
					                                                       sqrt(mymodel.sigma2_rot), sqrt(mymodel.sigma2_tilt), sqrt(mymodel.sigma2_psi));

					long int nr_orients = sampling.NrDirections() * sampling.NrPsiSamplings();
					if (nr_orients == 0)
					{
						std::cerr << " sampling.NrDirections()= " << sampling.NrDirections() << " sampling.NrPsiSamplings()= " << sampling.NrPsiSamplings() << std::endl;
						REPORT_ERROR("Zero orientations fall within the local angular search. Increase the sigma-value(s) on the orientations!");
					}

					int threadBlockSize = (nr_orients > 100) ? 10 : 1;

					exp_iorient_ThreadTaskDistributor->resize(nr_orients, threadBlockSize);

					global_mutex.unlock();

				}

				// Unpack the image from the imagedata
				img().resize(mymodel.ori_size, mymodel.ori_size);
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(img())
				{
					DIRECT_A2D_ELEM(img(), i, j) = DIRECT_A3D_ELEM(exp_imagedata, my_image_no, i, j);
				}
				img().setXmippOrigin();
				if (has_converged && do_use_reconstruct_images)
				{
					rec_img().resize(mymodel.ori_size, mymodel.ori_size);
					FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(rec_img())
					{
						DIRECT_A2D_ELEM(rec_img(), i, j) = DIRECT_A3D_ELEM(exp_imagedata, exp_nr_images + my_image_no, i, j);
					}
					rec_img().setXmippOrigin();
				}
				//#define DEBUG_SOFTMASK
#ifdef DEBUG_SOFTMASK
				Image<DOUBLE> tt;
				tt() = img();
				tt.write("Fimg_unmasked.spi");
				std::cerr << "written Fimg_unmasked.spi; press any key to continue..." << std::endl;
				char c;
				std::cin >> c;
#endif
				// Apply the norm_correction term
				if (do_norm_correction)
				{
					//#define DEBUG_NORM
#ifdef DEBUG_NORM
					if (normcorr < 0.001 || normcorr > 1000. || mymodel.avg_norm_correction < 0.001 || mymodel.avg_norm_correction > 1000.)
					{
						std::cerr << " ** normcorr= " << normcorr << std::endl;
						std::cerr << " ** mymodel.avg_norm_correction= " << mymodel.avg_norm_correction << std::endl;
						std::cerr << " ** fn_img= " << fn_img << " part_id= " << part_id << std::endl;
						std::cerr << " ** iseries= " << iseries << " ipart= " << ipart << " part_id= " << part_id << std::endl;
						int group_id = mydata.getGroupId(part_id, iseries);
						std::cerr << " ml_model.sigma2_noise[group_id]= " << mymodel.sigma2_noise[group_id] << " group_id= " << group_id << std::endl;
						std::cerr << " part_id= " << part_id << " iseries= " << iseries << std::endl;
						std::cerr << " img_id= " << img_id << std::endl;
						REPORT_ERROR("Very small or very big (avg) normcorr!");
					}
#endif
					img() *= mymodel.avg_norm_correction / normcorr;
				}

				// Apply (rounded) old offsets first
				my_old_offset.selfROUND();
				//for(int i =0; i < )
				selfTranslate(img(), my_old_offset, DONT_WRAP);

				if (has_converged && do_use_reconstruct_images)
				{
					selfTranslate(rec_img(), my_old_offset, DONT_WRAP);
				}

				exp_old_offset.at(my_image_no) = my_old_offset;
				// Also store priors on translations
				exp_prior.at(my_image_no) = my_prior;

				// Always store FT of image without mask (to be used for the reconstruction)
				MultidimArray<DOUBLE> img_aux;
				img_aux = (has_converged && do_use_reconstruct_images) ? rec_img() : img();
				CenterFFT(img_aux, true);
				transformer.FourierTransform(img_aux, Faux);
				windowFourierTransform(Faux, Fimg, mymodel.current_size);

				// Here apply the beamtilt correction if necessary
				// This will only be used for reconstruction, not for alignment
				// But beamtilt only affects very high-resolution components anyway...
				//
				DOUBLE beamtilt_x = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_BEAMTILT_X);
				DOUBLE beamtilt_y = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_BEAMTILT_Y);
				DOUBLE Cs = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_CS);
				DOUBLE V = 1000. * DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_VOLTAGE);
				DOUBLE lambda = 12.2643247 / sqrt(V * (1. + V * 0.978466e-6));
				if (ABS(beamtilt_x) > 0. || ABS(beamtilt_y) > 0.)
				{
					selfApplyBeamTilt(Fimg, beamtilt_x, beamtilt_y, lambda, Cs, mymodel.pixel_size, mymodel.ori_size);
				}

				exp_Fimgs_nomask.at(my_image_no) = Fimg;

				long int ori_part_id = exp_ipart_to_ori_part_id[ipart];

				MultidimArray<DOUBLE> Mnoise;

				// do we need to go through this pass?? the difficult is to process the random_seed generator
				// the current parallel program will not conside this pass
				if (!do_zero_mask)
				{
					// Make a noisy background image with the same spectrum as the sigma2_noise

					// Different MPI-distributed subsets may otherwise have different instances of the random noise below,
					// because work is on an on-demand basis and therefore variable with the timing of distinct nodes...
					// Have the seed based on the ipart, so that each particle has a different instant of the noise
					// Do this all inside a mutex for the threads, because they all use the same static variables inside ran1...
					// (So the mutex only goal is to make things exactly reproducible with the same random_seed.)
					global_mutex.lock();

					//init_random_generator(random_seed + ori_part_id);
					if (do_realign_movies)
					{
						init_random_generator(random_seed + part_id);
					}
					else
					{
						init_random_generator(random_seed + ori_part_id);
					}

					// If we're doing running averages, then the sigma2_noise was already adjusted for the running averages.
					// Undo this adjustment here in order to get the right noise in the individual frames
					MultidimArray<DOUBLE> power_noise = sigma2_fudge * mymodel.sigma2_noise[group_id];
					if (do_realign_movies)
					{
						power_noise *= (2. * movie_frame_running_avg_side + 1.);
					}

					// Create noisy image for outside the mask
					MultidimArray<Complex > Fnoise;
					Mnoise.resize(img());
					transformer.setReal(Mnoise);
					transformer.getFourierAlias(Fnoise);
					// Fill Fnoise with random numbers, use power spectrum of the noise for its variance
					FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fnoise)
					{
						int ires = ROUND(sqrt((DOUBLE)(kp * kp + ip * ip + jp * jp)));
						if (ires >= 0 && ires < XSIZE(Fnoise))
						{
							DOUBLE sigma = sqrt(DIRECT_A1D_ELEM(power_noise, ires));
							DIRECT_A3D_ELEM(Fnoise, k, i, j).real = rnd_gaus(0., sigma);
							DIRECT_A3D_ELEM(Fnoise, k, i, j).imag = rnd_gaus(0., sigma);
						}
						else
						{
							DIRECT_A3D_ELEM(Fnoise, k, i, j) = 0.;
						}
					}
					// Back to real space Mnoise
					transformer.inverseFourierTransform();
					Mnoise.setXmippOrigin();

					// unlock the mutex now that all calss to random functions have finished
					global_mutex.unlock();

					softMaskOutsideMap(img(), particle_diameter / (2. * mymodel.pixel_size), (DOUBLE)width_mask_edge, &Mnoise);

				}
				else
				{
					softMaskOutsideMap(img(), particle_diameter / (2. * mymodel.pixel_size), (DOUBLE)width_mask_edge);
				}
#ifdef DEBUG_SOFTMASK
				tt() = img();
				tt.write("Fimg_masked.spi");
				std::cerr << "written Fimg_masked.spi; press any key to continue..." << std::endl;
				exit(1);
				std::cin >> c;
#endif

				// Inside Projector and Backprojector the origin of the Fourier Transform is centered!
				CenterFFT(img(), true);

				// Store the Fourier Transform of the image Fimg
				transformer.FourierTransform(img(), Faux);

				// Store the power_class spectrum of the whole image (to fill sigma2_noise between current_size and ori_size
				if (mymodel.current_size < mymodel.ori_size)
				{
					MultidimArray<DOUBLE> spectrum;
					spectrum.initZeros(mymodel.ori_size / 2 + 1);
					DOUBLE highres_Xi2 = 0.;
					FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Faux)
					{
						int ires = ROUND(sqrt((DOUBLE)(kp * kp + ip * ip + jp * jp)));
						// Skip Hermitian pairs in the x==0 column

						if (ires > 0 && ires < mymodel.ori_size / 2 + 1 && !(jp == 0 && ip < 0))
						{
							DOUBLE normFaux = norm(DIRECT_A3D_ELEM(Faux, k, i, j));
							DIRECT_A1D_ELEM(spectrum, ires) += normFaux;
							// Store sumXi2 from current_size until ori_size
							if (ires >= mymodel.current_size / 2 + 1)
							{
								highres_Xi2 += normFaux;
							}
						}
					}

					// Let's use .at() here instead of [] to check whether we go outside the vectors bounds
					exp_power_imgs.at(my_image_no) = spectrum;
					exp_highres_Xi2_imgs.at(my_image_no) = highres_Xi2;
				}
				else
				{
					exp_highres_Xi2_imgs.at(my_image_no) = 0.;
				}

				// We never need any resolutions higher than current_size
				// So resize the Fourier transforms
				windowFourierTransform(Faux, Fimg, mymodel.current_size);

				// Also store its CTF
				Fctf.resize(Fimg);

				// Now calculate the actual CTF
				if (do_ctf_correction)
				{
					CTF ctf;
					ctf.setValues(DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_DEFOCUS_U),
					              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_DEFOCUS_V),
					              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_DEFOCUS_ANGLE),
					              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_VOLTAGE),
					              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_CS),
					              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_Q0),
					              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_BFAC));

					ctf.getFftwImage(Fctf, mymodel.ori_size, mymodel.ori_size, mymodel.pixel_size,
					                 ctf_phase_flipped, only_flip_phases, intact_ctf_first_peak, true);
					//#define DEBUG_CTF_FFTW_IMAGE
#ifdef DEBUG_CTF_FFTW_IMAGE
					Image<DOUBLE> tt;
					tt() = Fctf;
					tt.write("relion_ctf.spi");
					std::cerr << "Written relion_ctf.spi, now exiting..." << std::endl;
					exit(1);
#endif
					//#define DEBUG_GETCTF
#ifdef DEBUG_GETCTF
					std::cerr << " intact_ctf_first_peak= " << intact_ctf_first_peak << std::endl;
					ctf.write(std::cerr);
					Image<DOUBLE> tmp;
					tmp() = Fctf;
					tmp.write("Fctf.spi");
					tmp().resize(mymodel.ori_size, mymodel.ori_size);
					ctf.getCenteredImage(tmp(), mymodel.pixel_size, ctf_phase_flipped, only_flip_phases, intact_ctf_first_peak, true);
					tmp.write("Fctf_cen.spi");
					std::cerr << "Written Fctf.spi, Fctf_cen.spi. Press any key to continue..." << std::endl;
					char c;
					std::cin >> c;
#endif
				}
				else
				{
					Fctf.initConstant(1.);
				}

				// Store Fimg and Fctf
				exp_Fimgs.at(my_image_no) = Fimg;
				//HERE!!
				/*
				                std::cout << "doThreadGetFourierTransformsAndCtfs " << my_image_no << " ";
				                for (int i=0; i<10; i++)
				                    std::cout << exp_Fimgs.at(my_image_no).data[i].real << " " << exp_Fimgs.at(my_image_no).data[i].imag << " ";
				                std::cout << std::endl;
				*/
				exp_Fctfs.at(my_image_no) = Fctf;

			} // end loop iseries
		}// end loop ipart
	} // end while threadTaskDistributor

	// All threads clear out their transformer object when they are finished
	// This is to prevent a call from the first thread to fftw_cleanup, while there are still active plans in the transformer objects....
	// The multi-threaded code with FFTW objects is really a bit of a pain...
	if (thread_id != 0)
	{
		transformer.clear();
	}

	// Wait until all threads have finished
	global_barrier->wait();

	// Only the first thread cleans up the fftw-junk in the transformer object
	if (thread_id == 0)
	{
		transformer.cleanup();
	}

}

void MlOptimiser::doThreadPrecalculateShiftedImagesCtfsAndInvSigma2s(int thread_id)
{
#ifdef TIMING
	timer.tic(TIMING_DIFF_SHIFT);
#endif


	size_t first_ipart = 0, last_ipart = 0;
	while (exp_ipart_ThreadTaskDistributor->getTasks(first_ipart, last_ipart))
	{
		//std::cout << thread_id << " " << first_ipart << " " << last_ipart << "|";
		for (long int ipart = first_ipart; ipart <= last_ipart; ipart++)
		{
			// the exp_ipart_ThreadTaskDistributor was set with nr_pool,
			// but some, e.g. the last, batch of pooled particles may be smaller
			if (ipart >= exp_nr_particles)
			{
				break;
			}

			long int part_id = exp_ipart_to_part_id[ipart];
#ifdef DEBUG_CHECKSIZES
			if (ipart >= exp_starting_image_no.size())
			{
				std::cerr << "ipart= " << ipart << " exp_starting_image_no.size()= " << exp_starting_image_no.size() << std::endl;
				REPORT_ERROR("ipart >= exp_starting_image_no.size()");
			}
#endif
			int my_image_no = exp_starting_image_no[ipart] + exp_iseries;

#ifdef DEBUG_CHECKSIZES
			if (my_image_no >= exp_Fimgs.size())
			{
				std::cerr << "my_image_no= " << my_image_no << " exp_Fimgs.size()= " << exp_Fimgs.size() << std::endl;
				std::cerr << " exp_nr_trans= " << exp_nr_trans << " exp_nr_oversampled_trans= " << exp_nr_oversampled_trans << " exp_current_oversampling= " << exp_current_oversampling << std::endl;
				REPORT_ERROR("my_image_no >= exp_Fimgs.size()");
			}
#endif
			// Downsize Fimg and Fctf (again) to exp_current_image_size, also initialise Fref and Fimg_shift to the right size
			MultidimArray<Complex > Fimg, Fshifted, Fimg_nomask, Fshifted_nomask;
			windowFourierTransform(exp_Fimgs[my_image_no], Fimg, exp_current_image_size);
			windowFourierTransform(exp_Fimgs_nomask[my_image_no], Fimg_nomask, exp_current_image_size);


			// Also precalculate the sqrt of the sum of all Xi2
			// (Could exp_current_image_size ever be different from mymodel.current_size? Probhably therefore do it here rather than in getFourierTransforms
			if ((iter == 1 && do_firstiter_cc) || do_always_cc)
			{
				DOUBLE sumxi2 = 0.;
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fimg)
				{
					sumxi2 += norm(DIRECT_MULTIDIM_ELEM(Fimg, n));
				}
				// Normalised cross-correlation coefficient: divide by power of reference (power of image is a constant)
				exp_local_sqrtXi2[my_image_no] = sqrt(sumxi2);
			}

			// Store all translated variants of Fimg
			int my_trans_image = my_image_no * exp_nr_trans * exp_nr_oversampled_trans;
			for (long int itrans = 0; itrans < exp_nr_trans; itrans++)
			{
				// First get the non-oversampled translations as defined by the sampling object
				std::vector<Matrix1D <DOUBLE> > oversampled_translations;
				sampling.getTranslations(itrans, exp_current_oversampling, oversampled_translations);

#ifdef DEBUG_CHECKSIZES
				if (oversampled_translations.size() != exp_nr_oversampled_trans)
				{
					std::cerr << "oversampled_translations.size()= " << oversampled_translations.size() << " exp_nr_oversampled_trans= " << exp_nr_oversampled_trans << std::endl;
					REPORT_ERROR("oversampled_translations.size() != exp_nr_oversampled_trans");
				}
#endif

				// Then loop over all its oversampled relatives
				for (long int iover_trans = 0; iover_trans < exp_nr_oversampled_trans; iover_trans++)
				{
					//#define DEBUG_SHIFTS
#ifdef DEBUG_SHIFTS
					Image<DOUBLE> It;
					std::cerr << " iover_trans= " << iover_trans << " XX(oversampled_translations[iover_trans] )= " << XX(oversampled_translations[iover_trans]) << " YY(oversampled_translations[iover_trans] )= " << YY(oversampled_translations[iover_trans]) << std::endl;
#endif
					// Shift through phase-shifts in the Fourier transform
					// Note that the shift search range is centered around (exp_old_xoff, exp_old_yoff)
					shiftImageInFourierTransform(Fimg, Fshifted, tab_sin, tab_cos, (DOUBLE)mymodel.ori_size, oversampled_translations[iover_trans]);
					shiftImageInFourierTransform(Fimg_nomask, Fshifted_nomask, tab_sin, tab_cos, (DOUBLE)mymodel.ori_size, oversampled_translations[iover_trans]);

#ifdef DEBUG_SHIFTS
					FourierTransformer transformer;
					It().resize(YSIZE(Fimg), YSIZE(Fimg));
					transformer.inverseFourierTransform(Fimg, It());
					CenterFFT(It(), false);
					It.write("Fimg.spi");
					transformer.inverseFourierTransform(Fshifted, It());
					CenterFFT(It(), false);
					It.write("Fshifted.spi");
					std::cerr << "Written Fimg and Fshifted, press any key to continue..." << std::endl;
					char c;
					std::cin >> c;
#endif
					// Store the shifted image
					exp_local_Fimgs_shifted[my_trans_image] = Fshifted;
					exp_local_Fimgs_shifted_nomask[my_trans_image] = Fshifted_nomask;
					my_trans_image++;
				}
			}


			// Also store downsized Fctfs
			// In the second pass of the adaptive approach this will have no effect,
			// since then exp_current_image_size will be the same as the size of exp_Fctfs
#ifdef DEBUG_CHECKSIZES
			if (my_image_no >= exp_Fctfs.size())
			{
				std::cerr << "my_image_no= " << my_image_no << " exp_Fctfs.size()= " << exp_Fctfs.size() << std::endl;
				REPORT_ERROR("my_image_no >= exp_Fctfs.size()");
			}
#endif

			MultidimArray<DOUBLE> Fctf;
			windowFourierTransform(exp_Fctfs[my_image_no], Fctf, exp_current_image_size);
			exp_local_Fctfs[my_image_no] = Fctf;

			// Get micrograph id (for choosing the right sigma2_noise)
			int group_id = mydata.getGroupId(part_id, exp_iseries);

			MultidimArray<DOUBLE> Minvsigma2;
			Minvsigma2.initZeros(YSIZE(Fimg), XSIZE(Fimg));
			MultidimArray<int>* myMresol = (YSIZE(Fimg) == coarse_size) ? &Mresol_coarse : &Mresol_fine;

#ifdef DEBUG_CHECKSIZES
			if (!Minvsigma2.sameShape(*myMresol))
			{
				std::cerr << "!Minvsigma2.sameShape(*myMresol)= " << !Minvsigma2.sameShape(*myMresol) << std::endl;
				REPORT_ERROR("!Minvsigma2.sameShape(*myMresol)");
			}
#endif
			// With group_id and relevant size of Fimg, calculate inverse of sigma^2 for relevant parts of Mresol
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(*myMresol)
			{
				int ires = DIRECT_MULTIDIM_ELEM(*myMresol, n);
				// Exclude origin (ires==0) from the Probability-calculation
				// This way we are invariant to additive factors
				if (ires > 0)
				{
					DIRECT_MULTIDIM_ELEM(Minvsigma2, n) = 1. / (sigma2_fudge * DIRECT_A1D_ELEM(mymodel.sigma2_noise[group_id], ires));
				}
			}

#ifdef DEBUG_CHECKSIZES
			if (my_image_no >= exp_local_Minvsigma2s.size())
			{
				std::cerr << "my_image_no= " << my_image_no << " exp_local_Minvsigma2s.size()= " << exp_local_Minvsigma2s.size() << std::endl;
				REPORT_ERROR("my_image_no >= exp_local_Minvsigma2s.size()");
			}
#endif
			exp_local_Minvsigma2s[my_image_no] = Minvsigma2;
		}
	}

	// Wait until all threads are finsished
	global_barrier->wait();

#ifdef TIMING
	timer.toc(TIMING_DIFF_SHIFT);
#endif

}

bool MlOptimiser::isSignificantAnyParticleAnyTranslation(long int iorient)
{

	for (long int ipart = 0; ipart < YSIZE(exp_Mcoarse_significant); ipart++)
	{
		long int ihidden = iorient * exp_nr_trans;
		for (long int itrans = 0; itrans < exp_nr_trans; itrans++, ihidden++)
		{
#ifdef DEBUG_CHECKSIZES
			if (ihidden >= XSIZE(exp_Mcoarse_significant))
			{
				std::cerr << " ihidden= " << ihidden << " XSIZE(exp_Mcoarse_significant)= " << XSIZE(exp_Mcoarse_significant) << std::endl;
				std::cerr << " iorient= " << iorient << " itrans= " << itrans << " exp_nr_trans= " << exp_nr_trans << std::endl;
				REPORT_ERROR("ihidden > XSIZE: ");
			}
#endif
			if (DIRECT_A2D_ELEM(exp_Mcoarse_significant, ipart, ihidden))
			{
				return true;
			}
		}
	}
	return false;

}

void MlOptimiser::doThreadGetSquaredDifferencesAllOrientations(int thread_id)
{
	int tttt = 0;
#ifdef DEBUG_THREAD
	std::cerr << "entering doThreadGetAllSquaredDifferences" << std::endl;
#endif
	// Local variables

	std::vector<DOUBLE> thisthread_min_diff2;
	std::vector< Matrix1D<DOUBLE> > oversampled_orientations, oversampled_translations;
	MultidimArray<Complex > Fimg, Fref, Frefctf, Fimg_shift;
	MultidimArray<DOUBLE> Fctf, Minvsigma2;
	Matrix2D<DOUBLE> A;

	// Initialise local mindiff2 for thread-safety
	thisthread_min_diff2.clear();
	thisthread_min_diff2.resize(exp_nr_particles, 99.e99);
	Fref.resize(exp_local_Fimgs_shifted[0]);
	Frefctf.resize(exp_local_Fimgs_shifted[0]);

	// THESE TWO FOR LOOPS WILL BE PARALLELISED USING THREADS...
	// exp_iclass loop does not always go from 0 to nr_classes!
	long int iorientclass_offset = exp_iclass * exp_nr_rot;

	//  std::cout << exp_my_first_ori_particle << " " << exp_my_last_ori_particle << std::endl;
	size_t first_iorient = 0, last_iorient = 0;
	while (exp_iorient_ThreadTaskDistributor->getTasks(first_iorient, last_iorient))
	{
		for (long int iorient = first_iorient; iorient <= last_iorient; iorient++)
		{

			long int iorientclass = iorientclass_offset + iorient;
			long int idir = iorient / exp_nr_psi;
			long int ipsi = iorient % exp_nr_psi;
			// Get prior for this direction and skip calculation if prior==0
			DOUBLE pdf_orientation;
			if (mymodel.orientational_prior_mode == NOPRIOR)
			{
#ifdef DEBUG_CHECKSIZES
				if (idir >= XSIZE(mymodel.pdf_direction[exp_iclass]))
				{
					std::cerr << "idir= " << idir << " XSIZE(mymodel.pdf_direction[exp_iclass])= " << XSIZE(mymodel.pdf_direction[exp_iclass]) << std::endl;
					REPORT_ERROR("idir >= mymodel.pdf_direction[exp_iclass].size()");
				}
#endif
				pdf_orientation = DIRECT_MULTIDIM_ELEM(mymodel.pdf_direction[exp_iclass], idir);
			}
			else
			{
				pdf_orientation = sampling.getPriorProbability(idir, ipsi);
			}

			// In the first pass, always proceed
			// In the second pass, check whether one of the translations for this orientation of any of the particles had a significant weight in the first pass
			// if so, proceed with projecting the reference in that direction
			bool do_proceed = (exp_ipass == 0) ? true : isSignificantAnyParticleAnyTranslation(iorientclass);

			if (do_proceed && pdf_orientation > 0.)
			{
				// Now get the oversampled (rot, tilt, psi) triplets
				// This will be only the original (rot,tilt,psi) triplet in the first pass (exp_current_oversampling==0)
				sampling.getOrientations(idir, ipsi, exp_current_oversampling, oversampled_orientations);

#ifdef DEBUG_CHECKSIZES
				if (exp_nr_oversampled_rot != oversampled_orientations.size())
				{
					std::cerr << "exp_nr_oversampled_rot= " << exp_nr_oversampled_rot << " oversampled_orientations.size()= " << oversampled_orientations.size() << std::endl;
					REPORT_ERROR("exp_nr_oversampled_rot != oversampled_orientations.size()");
				}
#endif
				// Loop over all oversampled orientations (only a single one in the first pass)
				for (long int iover_rot = 0; iover_rot < exp_nr_oversampled_rot; iover_rot++)
				{

					// Get the Euler matrix
					Euler_angles2matrix(XX(oversampled_orientations[iover_rot]),
					                    YY(oversampled_orientations[iover_rot]),
					                    ZZ(oversampled_orientations[iover_rot]), A);

					// Take tilt-series into account
					A = (exp_R_mic * A).inv();

					// Project the reference map (into Fref)
#ifdef TIMING
					// Only time one thread, as I also only time one MPI process
					if (thread_id == 0)
					{
						timer.tic(TIMING_DIFF_PROJ);
					}
#endif
					(mymodel.PPref[exp_iclass]).get2DFourierTransform(Fref, A, IS_INV);

#ifdef TIMING
					// Only time one thread, as I also only time one MPI process
					if (thread_id == 0)
					{
						timer.toc(TIMING_DIFF_PROJ);
					}
#endif

					/// Now that reference projection has been made loop over someParticles!
					for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
					{
						// loop over all particles inside this ori_particle
						for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
						{
							long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];

							bool is_last_image_in_series = mydata.getNrImagesInSeries(part_id) == (exp_iseries + 1);
							// Which number was this image in the combined array of exp_iseries and part_id
							long int my_image_no = exp_starting_image_no[ipart] + exp_iseries;

#ifdef DEBUG_CHECKSIZES
							if (my_image_no >= exp_local_Minvsigma2s.size())
							{
								std::cerr << "my_image_no= " << my_image_no << " exp_local_Minvsigma2s.size()= " << exp_local_Minvsigma2s.size() << std::endl;
								REPORT_ERROR("my_image_no >= exp_local_Minvsigma2.size()");
							}
#endif
							Minvsigma2 = exp_local_Minvsigma2s[my_image_no];

							// Apply CTF to reference projection
							if (do_ctf_correction && refs_are_ctf_corrected)
							{

#ifdef DEBUG_CHECKSIZES
								if (my_image_no >= exp_local_Fctfs.size())
								{
									std::cerr << "my_image_no= " << my_image_no << " exp_local_Fctfs.size()= " << exp_local_Fctfs.size() << std::endl;
									REPORT_ERROR("my_image_no >= exp_local_Fctfs.size()");
								}
								if (MULTIDIM_SIZE(Fref) != MULTIDIM_SIZE(exp_local_Fctfs[my_image_no]))
								{
									std::cerr << "MULTIDIM_SIZE(Fref)= " << MULTIDIM_SIZE(Fref) << " MULTIDIM_SIZE()= " << MULTIDIM_SIZE(exp_local_Fctfs[my_image_no]) << std::endl;
									REPORT_ERROR("MULTIDIM_SIZE(Fref) != MULTIDIM_SIZE(exp_local_Fctfs[my_image_no)");
								}

#endif
								FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fref)
								{
									DIRECT_MULTIDIM_ELEM(Frefctf, n) = DIRECT_MULTIDIM_ELEM(Fref, n) * DIRECT_MULTIDIM_ELEM(exp_local_Fctfs[my_image_no], n);
								}
							}
							else
							{
								Frefctf = Fref;
							}

							if (do_scale_correction)
							{
								int group_id = mydata.getGroupId(part_id, exp_iseries);
#ifdef DEBUG_CHECKSIZES
								if (group_id >= mymodel.scale_correction.size())
								{
									std::cerr << "group_id= " << group_id << " mymodel.scale_correction.size()= " << mymodel.scale_correction.size() << std::endl;
									REPORT_ERROR("group_id >= mymodel.scale_correction.size()");
								}
#endif
								DOUBLE myscale = mymodel.scale_correction[group_id];
								FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Frefctf)
								{
									DIRECT_MULTIDIM_ELEM(Frefctf, n) *= myscale;
								}
							}

							for (long int itrans = 0; itrans < exp_nr_trans; itrans++)
							{
								long int ihidden = iorientclass * exp_nr_trans + itrans;

#ifdef DEBUG_CHECKSIZES
								if (exp_ipass > 0 && ihidden >= XSIZE(exp_Mcoarse_significant))
								{
									std::cerr << "ihidden= " << ihidden << " XSIZE(exp_Mcoarse_significant)= " << XSIZE(exp_Mcoarse_significant) << std::endl;
									REPORT_ERROR("ihidden >= XSIZE(exp_Mcoarse_significant)");
								}
#endif
								// In the first pass, always proceed
								// In the second pass, check whether this translations (&orientation) had a significant weight in the first pass
								bool do_proceed = (exp_ipass == 0) ? true : DIRECT_A2D_ELEM(exp_Mcoarse_significant, ipart, ihidden);
								if (do_proceed)
								{

									sampling.getTranslations(itrans, exp_current_oversampling, oversampled_translations);
									for (long int iover_trans = 0; iover_trans < exp_nr_oversampled_trans; iover_trans++)
									{
#ifdef TIMING
										// Only time one thread, as I also only time one MPI process
										if (thread_id == 0)
										{
											timer.tic(TIMING_DIFF_DIFF2);
										}
#endif
										// Get the shifted image
										long int ishift = my_image_no * exp_nr_oversampled_trans * exp_nr_trans +
										                  itrans * exp_nr_oversampled_trans + iover_trans;

#ifdef DEBUG_CHECKSIZES
										if (ishift >= exp_local_Fimgs_shifted.size())
										{
											std::cerr << "ishift= " << ishift << " exp_local_Fimgs_shifted.size()= " << exp_local_Fimgs_shifted.size() << std::endl;
											std::cerr << " itrans= " << itrans << std::endl;
											std::cerr << " ipart= " << ipart << std::endl;
											std::cerr << " exp_nr_oversampled_trans= " << exp_nr_oversampled_trans << " exp_nr_trans= " << exp_nr_trans << " iover_trans= " << iover_trans << std::endl;
											REPORT_ERROR("ishift >= exp_local_Fimgs_shifted.size()");
										}
#endif

										Fimg_shift = exp_local_Fimgs_shifted[ishift];

										//#define DEBUG_GETALLDIFF2
#ifdef DEBUG_GETALLDIFF2
										if (verb > 0)
										{
											FourierTransformer transformer;
											Image<DOUBLE> tt;
											tt().resize(exp_current_image_size, exp_current_image_size);
											transformer.inverseFourierTransform(Fimg_shift, tt());
											CenterFFT(tt(), false);
											tt.write("Fimg_shift.spi");
											transformer.inverseFourierTransform(Frefctf, tt());
											CenterFFT(tt(), false);
											tt.write("Fref.spi");
											tt() = Minvsigma2;
											tt.write("Minvsigma2.spi");
											std::cerr << "written Minvsigma2.spi" << std::endl;

											char c;
											std::cerr << "Written Fimg_shift.spi and Fref.spi. Press any key to continue..." << std::endl;
											std::cin >> c;
											exit(1);
										}
#endif

										DOUBLE diff2;
										if ((iter == 1 && do_firstiter_cc) || do_always_cc)
										{
											// Do not calculate squared-differences, but signal product
											// Negative values because smaller is worse in this case
											diff2 = 0.;
											DOUBLE suma2 = 0.;
											FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fimg_shift)
											{
												diff2 -= (DIRECT_MULTIDIM_ELEM(Frefctf, n)).real * (DIRECT_MULTIDIM_ELEM(Fimg_shift, n)).real;
												diff2 -= (DIRECT_MULTIDIM_ELEM(Frefctf, n)).imag * (DIRECT_MULTIDIM_ELEM(Fimg_shift, n)).imag;
												suma2 += norm(DIRECT_MULTIDIM_ELEM(Frefctf, n));
											}
											// Normalised cross-correlation coefficient: divide by power of reference (power of image is a constant)
											diff2 /= sqrt(suma2) * exp_local_sqrtXi2[my_image_no];
										}
										else
										{

#ifdef DEBUG_CHECKSIZES
											if (my_image_no >= exp_highres_Xi2_imgs.size())
											{
												std::cerr << "my_image_no= " << my_image_no << " exp_highres_Xi2_imgs.size()= " << exp_highres_Xi2_imgs.size() << std::endl;
												REPORT_ERROR("my_image_no >= exp_highres_Xi2_imgs.size()");
											}
#endif

											// Calculate the actual squared difference term of the Gaussian probability function
											// If current_size < mymodel.ori_size diff2 is initialised to the sum of
											// all |Xij|2 terms that lie between current_size and ori_size
											// Factor two because of factor 2 in division below, NOT because of 2-dimensionality of the complex plane!
											diff2 = exp_highres_Xi2_imgs[my_image_no] / 2.;
											FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fimg_shift)
											{
												DOUBLE diff_real = (DIRECT_MULTIDIM_ELEM(Frefctf, n)).real - (DIRECT_MULTIDIM_ELEM(Fimg_shift, n)).real;
												DOUBLE diff_imag = (DIRECT_MULTIDIM_ELEM(Frefctf, n)).imag - (DIRECT_MULTIDIM_ELEM(Fimg_shift, n)).imag;
												diff2 += (diff_real * diff_real + diff_imag * diff_imag) * 0.5 * DIRECT_MULTIDIM_ELEM(Minvsigma2, n);
											}
										}
										// if (tttt++ < 100) printf("%lf ", diff2);
										/*
										if (tttt++ < 21) {
										    printf("%3d # %lf cpu ", ishift, diff2);
										    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fimg_shift)
										    {
										        printf("%lf %lf ", DIRECT_MULTIDIM_ELEM(Fimg_shift, n).real, DIRECT_MULTIDIM_ELEM(Fimg_shift, n).imag);
										    }
										    printf("\n");
										    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fimg_shift)
										    {
										        printf("%lf %lf ", DIRECT_MULTIDIM_ELEM(Frefctf, n).real, DIRECT_MULTIDIM_ELEM(Frefctf, n).imag);
										    }
										    printf("\n");

										    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fimg_shift)
										    {
										        printf("%lf ", DIRECT_MULTIDIM_ELEM(Minvsigma2, n));
										    }

										    printf("\n");
										}
										*/
#ifdef TIMING
										// Only time one thread, as I also only time one MPI process
										if (thread_id == 0)
										{
											timer.toc(TIMING_DIFF_DIFF2);
										}
#endif

										// Store all diff2 in exp_Mweight
										long int ihidden_over = sampling.getPositionOversampledSamplingPoint(ihidden, exp_current_oversampling,
										                                                                     iover_rot, iover_trans);

										//#define DEBUG_DIFF2_ISNAN
#ifdef DEBUG_DIFF2_ISNAN
										if (std::isnan(diff2))
										{
											global_mutex.lock();
											std::cerr << " ipart= " << ipart << std::endl;
											std::cerr << " diff2= " << diff2 << " thisthread_min_diff2[ipart]= " << thisthread_min_diff2[ipart] << " ipart= " << ipart << std::endl;
											std::cerr << " exp_highres_Xi2_imgs[my_image_no]= " << exp_highres_Xi2_imgs[my_image_no] << std::endl;
											std::cerr << " exp_nr_oversampled_trans=" << exp_nr_oversampled_trans << std::endl;
											std::cerr << " exp_nr_oversampled_rot=" << exp_nr_oversampled_rot << std::endl;
											std::cerr << " iover_rot= " << iover_rot << " iover_trans= " << iover_trans << " ihidden= " << ihidden << std::endl;
											std::cerr << " exp_current_oversampling= " << exp_current_oversampling << std::endl;
											std::cerr << " ihidden_over= " << ihidden_over << " XSIZE(Mweight)= " << XSIZE(exp_Mweight) << std::endl;
											int group_id = mydata.getGroupId(part_id, exp_iseries);
											std::cerr << " mymodel.scale_correction[group_id]= " << mymodel.scale_correction[group_id] << std::endl;
											if (std::isnan(mymodel.scale_correction[group_id]))
											{
												for (int i = 0; i < mymodel.scale_correction.size(); i++)
												{
													std::cerr << " i= " << i << " mymodel.scale_correction[i]= " << mymodel.scale_correction[i] << std::endl;
												}
											}
											std::cerr << " group_id= " << group_id << std::endl;
											Image<DOUBLE> It;
											It() = Minvsigma2;
											It.write("Minvsigma2.spi");
											std::cerr << "written Minvsigma2.spi" << std::endl;
											std::cerr << "Frefctf shape= ";
											Frefctf.printShape(std::cerr);
											std::cerr << "Fimg_shift shape= ";
											Fimg_shift.printShape(std::cerr);
											It() = exp_local_Fctfs[my_image_no];
											It.write("exp_local_Fctf.spi");
											std::cerr << "written exp_local_Fctf.spi" << std::endl;
											FourierTransformer transformer;
											Image<DOUBLE> tt;
											tt().resize(exp_current_image_size, exp_current_image_size);
											transformer.inverseFourierTransform(Fimg_shift, tt());
											CenterFFT(tt(), false);
											tt.write("Fimg_shift.spi");
											std::cerr << "written Fimg_shift.spi" << std::endl;
											FourierTransformer transformer2;
											tt().initZeros();
											transformer2.inverseFourierTransform(Frefctf, tt());
											CenterFFT(tt(), false);
											tt.write("Frefctf.spi");
											std::cerr << "written Frefctf.spi" << std::endl;
											FourierTransformer transformer3;
											tt().initZeros();
											transformer3.inverseFourierTransform(Fref, tt());
											CenterFFT(tt(), false);
											tt.write("Fref.spi");
											std::cerr << "written Fref.spi" << std::endl;
											std::cerr << " A= " << A << std::endl;
											std::cerr << " exp_R_mic= " << exp_R_mic << std::endl;
											std::cerr << "written Frefctf.spi" << std::endl;
											REPORT_ERROR("diff2 is not a number");
											global_mutex.unlock();
										}
#endif
										//#define DEBUG_VERBOSE
#ifdef DEBUG_VERBOSE
										global_mutex.lock();
										if (verb > 0)
										{
											std::cout << " rot= " << XX(oversampled_orientations[iover_rot]) << " tilt= " << YY(oversampled_orientations[iover_rot]) << " psi= " << ZZ(oversampled_orientations[iover_rot]) << std::endl;
											std::cout << " xoff= " << XX(oversampled_translations[iover_trans]) << " yoff= " << YY(oversampled_translations[iover_trans]) << std::endl;
											std::cout << " ihidden_over= " << ihidden_over << " diff2= " << diff2 << " thisthread_min_diff2[ipart]= " << thisthread_min_diff2[ipart] << std::endl;
										}
										global_mutex.unlock();
#endif
#ifdef DEBUG_CHECKSIZES
										if (ihidden_over >= XSIZE(exp_Mweight))
										{
											std::cerr << " exp_nr_oversampled_trans=" << exp_nr_oversampled_trans << std::endl;
											std::cerr << " exp_nr_oversampled_rot=" << exp_nr_oversampled_rot << std::endl;
											std::cerr << " iover_rot= " << iover_rot << " iover_trans= " << iover_trans << " ihidden= " << ihidden << std::endl;
											std::cerr << " exp_current_oversampling= " << exp_current_oversampling << std::endl;
											std::cerr << " ihidden_over= " << ihidden_over << " XSIZE(Mweight)= " << XSIZE(exp_Mweight) << std::endl;
											REPORT_ERROR("ihidden_over >= XSIZE(Mweight)");
										}
#endif
										//std::cout << XSIZE(exp_Mweight) << ":" << iorient << " " << iover_rot << " " << itrans << " " << iover_trans << " " << ihidden_over << std::endl;

										if (exp_iseries == 0)
										{
											DIRECT_A2D_ELEM(exp_Mweight, ipart, ihidden_over) = diff2;
										}
										else
										{
											DIRECT_A2D_ELEM(exp_Mweight, ipart, ihidden_over) += diff2;
										}

#ifdef DEBUG_CHECKSIZES
										if (ipart >= thisthread_min_diff2.size())
										{
											std::cerr << "ipart= " << ipart << " thisthread_min_diff2.size()= " << thisthread_min_diff2.size() << std::endl;
											REPORT_ERROR("ipart >= thisthread_min_diff2.size() ");
										}
#endif
										// Keep track of minimum of all diff2, only for the last image in this series
										diff2 = DIRECT_A2D_ELEM(exp_Mweight, ipart, ihidden_over);
										//std::cerr << " exp_ipass= " << exp_ipass << " exp_iclass= " << exp_iclass << " diff2= " << diff2 << std::endl;
										if (is_last_image_in_series && diff2 < thisthread_min_diff2[ipart])
										{
											thisthread_min_diff2[ipart] = diff2;
										}

									} // end loop iover_trans
								} // end if do_proceed translations
							} // end loop itrans
						} // end loop part_id (i)
					} // end loop ori_part_id
				}// end loop iover_rot
			} // end if do_proceed orientations
		} // end loop iorient
	} // end while task distribution


	// Now inside a mutex set the minimum of the squared differences among all threads
#ifdef DEBUG_CHECKSIZES
	if (thisthread_min_diff2.size() != exp_min_diff2.size())
	{
		std::cerr << "thisthread_min_diff2.size()= " << thisthread_min_diff2.size() << " exp_min_diff2.size()= " << exp_min_diff2.size() << std::endl;
		REPORT_ERROR("thisthread_min_diff2.size() != exp_min_diff2.size()");
	}
#endif

	global_mutex.lock();
	for (int i = 0; i < exp_min_diff2.size(); i++)
	{
		if (thisthread_min_diff2[i] < exp_min_diff2[i])
		{
			exp_min_diff2[i] = thisthread_min_diff2[i];
		}
	}


	global_mutex.unlock();

	// Wait until all threads have finished
	global_barrier->wait();

#ifdef DEBUG_THREAD
	std::cerr << "leaving doThreadGetAllSquaredDifferences" << std::endl;
#endif



}


void MlOptimiser::getAllSquaredDifferences()
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

	// Initialise min_diff and exp_Mweight for this pass
	exp_Mweight.resize(exp_nr_particles, mymodel.nr_classes * sampling.NrSamplingPoints(exp_current_oversampling, false));
	exp_Mweight.initConstant(-999.);

	if (exp_ipass == 0)
	{
		exp_Mcoarse_significant.clear();
	}

	exp_min_diff2.clear();
	exp_min_diff2.resize(exp_nr_particles);
	for (int n = 0; n < exp_nr_particles; n++)
	{
		exp_min_diff2[n] = 99.e99;
	}

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

	// TODO: MAKE SURE THAT ALL PARTICLES IN SomeParticles ARE FROM THE SAME AREA, SO THAT THE R_mic CAN BE RE_USED!!!

	//for (exp_iseries = 0; exp_iseries < mydata.getNrImagesInSeries(part_id); exp_iseries++)
	for (exp_iseries = 0; exp_iseries < mydata.getNrImagesInSeries((mydata.ori_particles[exp_my_first_ori_particle]).particles_id[0]); exp_iseries++)
	{

		// Get all shifted versions of the (downsized) images, their (downsized) CTFs and their inverted Sigma2 matrices
		exp_ipart_ThreadTaskDistributor->reset(); // reset thread distribution tasks
		global_ThreadManager->run(globalThreadPrecalculateShiftedImagesCtfsAndInvSigma2s);
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

		// Loop from iclass_min to iclass_max to deal with seed generation in first iteration
		for (exp_iclass = iclass_min; exp_iclass <= iclass_max; exp_iclass++)
		{
			if (mymodel.pdf_class[exp_iclass] > 0.)
			{
				exp_iorient_ThreadTaskDistributor->reset(); // reset thread distribution tasks
				global_ThreadManager->run(globalThreadGetSquaredDifferencesAllOrientations);
			} // end if mymodel.pdf_class[iclass] > 0.
		} // end loop iclass
	} // end loop iseries
	for (int i = 0; i < exp_nr_particles; i++)
	{
		//exp_min_diff2[i] = mini_weight_particle[i];
		std::cout << "  " << exp_min_diff2[i] << "  " << (iclass_max - iclass_min + 1) * sampling.NrSamplingPoints(exp_current_oversampling, false);
	}
		std::cout << " exp_ipass = " << exp_ipass <<  "  iter = " << iter << std::endl;
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

void MlOptimiser::doThreadConvertSquaredDifferencesToWeightsAllOrientations(int thread_id)
{
#ifdef DEBUG_THREAD
	std::cerr << "entering doThreadConvertSquaredDifferencesToWeightsAllOrientations" << std::endl;
#endif


	// Store local sum of weights for this thread and then combined all threads at the end of this function inside a mutex.
	DOUBLE thisthread_sumweight = 0.;

	// exp_iclass loop does not always go from 0 to nr_classes!
	long int iorientclass_offset = exp_iclass * exp_nr_rot;

	size_t first_iorient = 0, last_iorient = 0;
	while (exp_iorient_ThreadTaskDistributor->getTasks(first_iorient, last_iorient))
	{
		for (long int iorient = first_iorient; iorient <= last_iorient; iorient++)
		{

			DOUBLE pdf_orientation;
			long int iorientclass = iorientclass_offset + iorient;
			long int idir = iorient / exp_nr_psi;
			long int ipsi = iorient % exp_nr_psi;

			// Get prior for this direction
			if (mymodel.orientational_prior_mode == NOPRIOR)
			{
#ifdef DEBUG_CHECKSIZES
				if (idir >= XSIZE(mymodel.pdf_direction[exp_iclass]))
				{
					std::cerr << "idir= " << idir << " XSIZE(mymodel.pdf_direction[exp_iclass])= " << XSIZE(mymodel.pdf_direction[exp_iclass]) << std::endl;
					REPORT_ERROR("idir >= mymodel.pdf_direction[exp_iclass].size()");
				}
#endif
				pdf_orientation = DIRECT_MULTIDIM_ELEM(mymodel.pdf_direction[exp_iclass], idir);
			}
			else
			{
				pdf_orientation = sampling.getPriorProbability(idir, ipsi);
			}

			// Loop over all translations
			for (long int itrans = 0; itrans < exp_nr_trans; itrans++)
			{

				long int ihidden = iorientclass * exp_nr_trans + itrans;

				// To speed things up, only calculate pdf_offset at the coarse sampling.
				// That should not matter much, and that way one does not need to calculate all the OversampledTranslations
				Matrix1D<DOUBLE> my_offset, my_prior;
				sampling.getTranslation(itrans, my_offset);
				// Convert offsets back to Angstroms to calculate PDF!
				// TODO: if series, then have different exp_old_xoff for each my_image_no....
				// WHAT TO DO WITH THIS?!!!

				DOUBLE pdf_offset;
				if (mymodel.ref_dim == 2)
				{
					pdf_offset = calculatePdfOffset(exp_old_offset[exp_iimage] + my_offset, mymodel.prior_offset_class[exp_iclass]);
				}
				else
				{
					pdf_offset = calculatePdfOffset(exp_old_offset[exp_iimage] + my_offset, exp_prior[exp_iimage]);
				}

				// TMP DEBUGGING
				if (mymodel.orientational_prior_mode != NOPRIOR && (pdf_offset == 0. || pdf_orientation == 0.))
				{
					global_mutex.lock();
					std::cerr << " pdf_offset= " << pdf_offset << " pdf_orientation= " << pdf_orientation << std::endl;
					std::cerr << " exp_ipart= " << exp_ipart << " exp_part_id= " << exp_part_id << std::endl;
					std::cerr << " iorient= " << iorient << " idir= " << idir << " ipsi= " << ipsi << std::endl;
					std::cerr << " exp_nr_psi= " << exp_nr_psi << " exp_nr_dir= " << exp_nr_dir << " exp_nr_trans= " << exp_nr_trans << std::endl;
					for (long int i = 0; i < sampling.directions_prior.size(); i++)
					{
						std::cerr << " sampling.directions_prior[" << i << "]= " << sampling.directions_prior[i] << std::endl;
					}
					for (long int i = 0; i < sampling.psi_prior.size(); i++)
					{
						std::cerr << " sampling.psi_prior[" << i << "]= " << sampling.psi_prior[i] << std::endl;
					}
					REPORT_ERROR("ERROR! pdf_offset==0.|| pdf_orientation==0.");
					global_mutex.unlock();
				}
				if (exp_nr_oversampled_rot == 0)
				{
					REPORT_ERROR("exp_nr_oversampled_rot == 0");
				}
				if (exp_nr_oversampled_trans == 0)
				{
					REPORT_ERROR("exp_nr_oversampled_trans == 0");
				}


#ifdef TIMING
				// Only time one thread, as I also only time one MPI process
				if (thread_id == 0)
				{
					timer.tic(TIMING_WEIGHT_EXP);
				}
#endif

				// Now first loop over iover_rot, because that is the order in exp_Mweight as well
				long int ihidden_over = ihidden * exp_nr_oversampled_rot * exp_nr_oversampled_trans;
				for (long int iover_rot = 0; iover_rot < exp_nr_oversampled_rot; iover_rot++)
				{
					// Then loop over iover_trans
					for (long int iover_trans = 0; iover_trans < exp_nr_oversampled_trans; iover_trans++, ihidden_over++)
					{



						// Only exponentiate for determined values of exp_Mweight
						// (this is always true in the first pass, but not so in the second pass)
						// Only deal with this sampling point if its weight was significant


						if (DIRECT_A2D_ELEM(exp_Mweight, exp_ipart, ihidden_over) < 0.)
						{
							DIRECT_A2D_ELEM(exp_Mweight, exp_ipart, ihidden_over) = 0.;
						}
						else
						{
							DOUBLE weight = pdf_orientation * pdf_offset;

							DOUBLE diff2 = DIRECT_A2D_ELEM(exp_Mweight, exp_ipart, ihidden_over) - exp_min_diff2[exp_ipart];

							// next line because of numerical precision of exp-function
							if (diff2 > 700.)
							{
								weight = 0.;
							}
							// TODO: use tabulated exp function?
							else
							{
								weight *= exp(-diff2);
							}

							// Store the weight
							DIRECT_A2D_ELEM(exp_Mweight, exp_ipart, ihidden_over) = weight;


							//                          std::cout << ihidden_over << " = " << exp_iclass << " " << iorient << " " << itrans << " " << iover_rot << " " << iover_trans << std::endl;

							//                          std::cout << ihidden_over << " ==" << iclass_max - iclass_min << " " << exp_nr_rot << " " << exp_nr_trans << " " << exp_nr_oversampled_rot << " " << exp_nr_oversampled_trans << std::endl;



							// Keep track of sum and maximum of all weights for this particle
							// Later add all to exp_thisparticle_sumweight, but inside this loop sum to local thisthread_sumweight first
							thisthread_sumweight += weight;

						} // end if/else exp_Mweight < 0.
					} // end loop iover_trans
				}// end loop iover_rot
#ifdef timing
				// only time one thread, as i also only time one mpi process
				if (thread_id == 0)
				{
					timer.toc(timing_weight_exp);
				}
#endif
			} // end loop itrans

		} // end loop iorient
	} // end while task distributor

	// Now inside a mutex update the sum of all weights
	global_mutex.lock();
	exp_thisparticle_sumweight += thisthread_sumweight;

	global_mutex.unlock();

	// Wait until all threads have finished
	global_barrier->wait();

	//  std::cout <<  thisthread_sumweight << " " << exp_thisparticle_sumweight << std::endl;


#ifdef DEBUG_THREAD
	std::cerr << "leaving doThreadConvertSquaredDifferencesToWeightsAllOrientations" << std::endl;
#endif
}

void MlOptimiser::convertAllSquaredDifferencesToWeights()
{

#ifdef TIMING
	if (exp_ipass == 0)
	{
		timer.tic(TIMING_ESP_WEIGHT1);
	}
	else
	{
		timer.tic(TIMING_ESP_WEIGHT2);
	}
#endif

	// Convert the squared differences into weights
	// Note there is only one weight for each part_id, because a whole series of images is treated as one particle

	// Initialising...
	exp_sum_weight.resize(exp_nr_particles);
	for (int i = 0; i < exp_nr_particles; i++)
	{
		exp_sum_weight[i] = 0.;
	}

	//#define DEBUG_CONVERTDIFF2W
#ifdef DEBUG_CONVERTDIFF2W
	DOUBLE max_weight = -1.;
	DOUBLE opt_psi, opt_xoff, opt_yoff;
	int opt_iover_rot, opt_iover_trans, opt_ipsi, opt_itrans;
	long int opt_ihidden, opt_ihidden_over;
#endif

	//TMP DEBUGGING
	//DEBUGGING_COPY_exp_Mweight = exp_Mweight;

	// Loop from iclass_min to iclass_max to deal with seed generation in first iteration
	exp_iimage = 0;
	exp_ipart = 0;
	for (long int ori_part_id = exp_my_first_ori_particle; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		// loop over all particles inside this ori_particle
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, exp_ipart++)
		{
			exp_part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			exp_thisparticle_sumweight = 0.;


			if ((iter == 1 && do_firstiter_cc) || do_always_cc)
			{

				// Binarize the squared differences array to skip marginalisation
				// Note this loop is not threaded. This is not so important because it will only be executed in the 1st iteration and is fast anyway
				DOUBLE mymindiff2 = 99.e10, mymaxprob = -99.e10;
				long int myminidx = -1;
				// Find the smallest element in this row of exp_Mweight
				for (long int i = 0; i < XSIZE(exp_Mweight); i++)
				{

					DOUBLE cc = DIRECT_A2D_ELEM(exp_Mweight, exp_ipart, i);
					// ignore non-determined cc
					if (cc == -999.)
					{
						continue;
					}

					if (do_sim_anneal && iter > 1)
					{
						// P_accept = exp ( - (CCold -CC)/temperature)
						// cc is negative value, so use "+ cc"
						DOUBLE my_prob = rnd_unif() * exp(-(exp_local_oldcc[exp_ipart] + cc) / temperature);
						if (my_prob > mymaxprob)
						{
							mymaxprob = my_prob;
							mymindiff2 = cc;
							myminidx = i;
						}
					}
					else
					{
						// just search for the maximum
						if (cc < mymindiff2)
						{
							mymindiff2 = cc;
							myminidx = i;
						}
					}
				}
				// Set all except for the best hidden variable to zero and the smallest element to 1
				for (long int i = 0; i < XSIZE(exp_Mweight); i++)
				{
					DIRECT_A2D_ELEM(exp_Mweight, exp_ipart, i) = 0.;
				}

				DIRECT_A2D_ELEM(exp_Mweight, exp_ipart, myminidx) = 1.;
				exp_thisparticle_sumweight += 1.;

			}
			else
			{
				//std::cout << "exp_iclass " << iclass_min << " " << iclass_max << std::endl;
				for (exp_iclass = iclass_min; exp_iclass <= iclass_max; exp_iclass++)
				{

					// The loops over all orientations are parallelised using threads
					exp_iorient_ThreadTaskDistributor->reset(); // reset thread distribution tasks
					global_ThreadManager->run(globalThreadConvertSquaredDifferencesToWeightsAllOrientations);

				} // end loop iclass

			} // end else iter==1 && do_firstiter_cc

			// Keep track of number of processed images
			exp_iimage += mydata.getNrImagesInSeries(exp_part_id);

			//Store parameters for this particle
			exp_sum_weight[exp_ipart] = exp_thisparticle_sumweight;

			// Check the sum of weights is not zero
			// On a Mac, the isnan function does not compile. Just uncomment the define statement, as this is merely a debugging statement
			//#define MAC_OSX
#ifndef MAC_OSX
			if (exp_thisparticle_sumweight == 0. || std::isnan(exp_thisparticle_sumweight))
			{
				std::cerr << " exp_thisparticle_sumweight= " << exp_thisparticle_sumweight << std::endl;
				Image<DOUBLE> It;
				It() = exp_Mweight;
				It.write("Mweight.spi");
				//It() = DEBUGGING_COPY_exp_Mweight;
				//It.write("Mweight_copy.spi");
				It().resize(exp_Mcoarse_significant);
				if (MULTIDIM_SIZE(It()) > 0)
				{
					FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(It())
					{
						if (DIRECT_MULTIDIM_ELEM(exp_Mcoarse_significant, n))
						{
							DIRECT_MULTIDIM_ELEM(It(), n) = 1.;
						}
						else
						{
							DIRECT_MULTIDIM_ELEM(It(), n) = 0.;
						}
					}
					It.write("Mcoarse_significant.spi");
				}
				std::cerr << " exp_part_id= " << exp_part_id << "exp_iimage=" << exp_iimage << std::endl;
				int group_id = mydata.getGroupId(exp_part_id, 0);
				std::cerr << " group_id= " << group_id << " mymodel.scale_correction[group_id]= " << mymodel.scale_correction[group_id] << std::endl;
				std::cerr << " exp_ipass= " << exp_ipass << std::endl;
				std::cerr << " sampling.NrDirections(0, true)= " << sampling.NrDirections(0, true)
				          << " sampling.NrDirections(0, false)= " << sampling.NrDirections(0, false) << std::endl;
				std::cerr << " sampling.NrPsiSamplings(0, true)= " << sampling.NrPsiSamplings(0, true)
				          << " sampling.NrPsiSamplings(0, false)= " << sampling.NrPsiSamplings(0, false) << std::endl;
				std::cerr << " mymodel.sigma2_noise[exp_ipart]= " << mymodel.sigma2_noise[exp_ipart] << std::endl;
				std::cerr << " wsum_model.sigma2_noise[exp_ipart]= " << wsum_model.sigma2_noise[exp_ipart] << std::endl;
				if (mymodel.orientational_prior_mode == NOPRIOR)
				{
					std::cerr << " wsum_model.pdf_direction[exp_ipart]= " << wsum_model.pdf_direction[exp_ipart] << std::endl;
				}
				if (do_norm_correction)
				{
					std::cerr << " mymodel.avg_norm_correction= " << mymodel.avg_norm_correction << std::endl;
					std::cerr << " wsum_model.avg_norm_correction= " << wsum_model.avg_norm_correction << std::endl;
				}

				std::cerr << "written out Mweight.spi" << std::endl;
				std::cerr << " exp_thisparticle_sumweight= " << exp_thisparticle_sumweight << std::endl;
				std::cerr << " exp_min_diff2[exp_ipart]= " << exp_min_diff2[exp_ipart] << std::endl;
				REPORT_ERROR("ERROR!!! zero sum of weights....");
			}
#endif

		} // end loop part_id (i)
	} // end loop ori_part_id
	/*
	std::cout << "exp_sum_weight = ";
	for (int i = 0; i < exp_nr_particles; i++)
	    std::cout << exp_sum_weight[i] << " ";
	std::cout << std::endl;
	*/
	// The remainder of this function is not threaded.

	// Initialise exp_Mcoarse_significant
	if (exp_ipass == 0)
	{
		exp_Mcoarse_significant.resize(exp_nr_particles, XSIZE(exp_Mweight));
	}

	// Now, for each particle,  find the exp_significant_weight that encompasses adaptive_fraction of exp_sum_weight
	exp_significant_weight.clear();
	exp_significant_weight.resize(exp_nr_particles, 0.);
	for (long int ori_part_id = exp_my_first_ori_particle, my_image_no = 0, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		// loop over all particles inside this ori_particle
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];

#ifdef TIMING
			timer.tic(TIMING_WEIGHT_SORT);
#endif
			MultidimArray<DOUBLE> sorted_weight;
			// Get the relevant row for this particle
			exp_Mweight.getRow(ipart, sorted_weight);

			// Only select non-zero probabilities to speed up sorting
			long int np = 0;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sorted_weight)
			{
				if (DIRECT_MULTIDIM_ELEM(sorted_weight, n) > 0.)
				{
					DIRECT_MULTIDIM_ELEM(sorted_weight, np) = DIRECT_MULTIDIM_ELEM(sorted_weight, n);
					np++;
				}
			}
			sorted_weight.resize(np);
			std::cout << " particle id  = " << ori_part_id   <<  " sorted_weight  = " << np <<  "  iter = " << iter << std::endl;
			// Sort from low to high values
			sorted_weight.sort();

#ifdef TIMING
			timer.toc(TIMING_WEIGHT_SORT);
#endif
			DOUBLE frac_weight = 0.;
			DOUBLE my_significant_weight;
			long int my_nr_significant_coarse_samples = 0;
			for (long int i = XSIZE(sorted_weight) - 1; i >= 0; i--)
			{
				if (exp_ipass == 0)
				{
					my_nr_significant_coarse_samples++;
				}
				my_significant_weight = DIRECT_A1D_ELEM(sorted_weight, i);
				frac_weight += my_significant_weight;
				if (frac_weight > adaptive_fraction * exp_sum_weight[ipart])
				{
					break;
				}
			}

#ifdef DEBUG_SORT
			// Check sorted array is really sorted
			DOUBLE prev = 0.;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sorted_weight)
			{
				if (DIRECT_MULTIDIM_ELEM(sorted_weight, n) < prev)
				{
					Image<DOUBLE> It;
					It() = sorted_weight;
					It() *= 10000;
					It.write("sorted_weight.spi");
					std::cerr << "written sorted_weight.spi" << std::endl;
					REPORT_ERROR("Error in sorting!");
				}
				prev = DIRECT_MULTIDIM_ELEM(sorted_weight, n);
			}
#endif

			if (exp_ipass == 0 && my_nr_significant_coarse_samples == 0)
			{
				std::cerr << " ipart= " << ipart << " adaptive_fraction= " << adaptive_fraction << std::endl;
				std::cerr << " frac-weight= " << frac_weight << std::endl;
				std::cerr << " exp_sum_weight[ipart]= " << exp_sum_weight[ipart] << std::endl;
				Image<DOUBLE> It;
				std::cerr << " XSIZE(exp_Mweight)= " << XSIZE(exp_Mweight) << std::endl;
				It() = exp_Mweight;
				It() *= 10000;
				It.write("Mweight2.spi");
				std::cerr << "written Mweight2.spi" << std::endl;
				std::cerr << " np= " << np << std::endl;
				It() = sorted_weight;
				It() *= 10000;
				std::cerr << " XSIZE(sorted_weight)= " << XSIZE(sorted_weight) << std::endl;
				if (XSIZE(sorted_weight) > 0)
				{
					It.write("sorted_weight.spi");
					std::cerr << "written sorted_weight.spi" << std::endl;
				}
				REPORT_ERROR("my_nr_significant_coarse_samples == 0");
			}

			if (exp_ipass == 0)
			{
				// Store nr_significant_coarse_samples for all images in this series
				for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++, my_image_no++)
				{
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NR_SIGN) = (DOUBLE)my_nr_significant_coarse_samples;
				}

				// Keep track of which coarse samplings were significant were significant for this particle
				for (int ihidden = 0; ihidden < XSIZE(exp_Mcoarse_significant); ihidden++)
				{
					if (DIRECT_A2D_ELEM(exp_Mweight, ipart, ihidden) >= my_significant_weight)
					{
						DIRECT_A2D_ELEM(exp_Mcoarse_significant, ipart, ihidden) = true;
					}
					else
					{
						DIRECT_A2D_ELEM(exp_Mcoarse_significant, ipart, ihidden) = false;
					}
				}

			}
			exp_significant_weight[ipart] = my_significant_weight;
#ifdef DEBUG_OVERSAMPLING
			std::cerr << " sum_weight[ipart]= " << exp_sum_weight[ipart] << " my_significant_weight= " << my_significant_weight << std::endl;
			std::cerr << " my_nr_significant_coarse_samples= " << my_nr_significant_coarse_samples << std::endl;
			std::cerr << " ipass= " << exp_ipass << " Pmax=" << DIRECT_A1D_ELEM(sorted_weight, XSIZE(sorted_weight) - 1) / frac_weight
			          << " nr_sign_sam= " << nr_significant_samples << " sign w= " << exp_significant_weight << "sum_weight= " << exp_sum_weight << std::endl;
#endif

		} // end loop part_id (i)
	} // end loop ori_part_id


#ifdef DEBUG_CONVERTDIFF2W
	//Image<DOUBLE> tt;
	//tt()=sorted_weight;
	//tt.write("sorted_weight.spi");
	//std::cerr << "written sorted_weight.spi" << std::endl;
	std::cerr << " ipass= " << exp_ipass << " exp_part_id= " << exp_part_id << std::endl;
	std::cerr << " diff2w: opt_xoff= " << opt_xoff << " opt_yoff= " << opt_yoff << " opt_psi= " << opt_psi << std::endl;
	std::cerr << " diff2w: opt_iover_rot= " << opt_iover_rot << " opt_iover_trans= " << opt_iover_trans << " opt_ipsi= " << opt_ipsi << std::endl;
	std::cerr << " diff2w: opt_itrans= " << opt_itrans << " opt_ihidden= " << opt_ihidden << " opt_ihidden_over= " << opt_ihidden_over << std::endl;
	std::cerr << "significant_weight= " << exp_significant_weight << " max_weight= " << max_weight << std::endl;
	std::cerr << "nr_significant_coarse_samples= " << nr_significant_coarse_samples << std::endl;
	debug2 = (DOUBLE)opt_ihidden_over;
#endif

#ifdef TIMING
	if (exp_ipass == 0)
	{
		timer.toc(TIMING_ESP_WEIGHT1);
	}
	else
	{
		timer.toc(TIMING_ESP_WEIGHT2);
	}
#endif

}

void MlOptimiser::doThreadStoreWeightedSumsAllOrientations(int thread_id)
{
#ifdef DEBUG_THREAD
	std::cerr << "entering doThreadStoreWeightedSumsAllOrientations" << std::endl;
#endif


	std::vector< Matrix1D<DOUBLE> > oversampled_orientations, oversampled_translations;
	Matrix2D<DOUBLE> A;
	MultidimArray<Complex > Fimg, Fref, Frefctf, Fimg_shift, Fimg_shift_nomask;
	MultidimArray<DOUBLE> Minvsigma2, Mctf, Fweight;
	DOUBLE rot, tilt, psi;
	bool have_warned_small_scale = false;

	// Initialising...
	Fref.resize(exp_Fimgs[0]);
	Frefctf.resize(exp_Fimgs[0]);
	Fweight.resize(exp_Fimgs[0]);

	// Initialise Mctf to all-1 for if !do_ctf_corection
	Mctf.resize(exp_Fimgs[0]);
	Mctf.initConstant(1.);

	// Initialise Minvsigma2 to all-1 for if !do_map
	Minvsigma2.resize(exp_Fimgs[0]);
	Minvsigma2.initConstant(1.);

	// Make local copies of weighted sums (excepts BPrefs, which are too big)
	// so that there are not too many mutex locks below
	std::vector<MultidimArray<DOUBLE> > thr_wsum_sigma2_noise, thr_wsum_scale_correction_XA, thr_wsum_scale_correction_AA, thr_wsum_pdf_direction;
	std::vector<DOUBLE> thr_wsum_norm_correction, thr_sumw_group, thr_wsum_pdf_class, thr_wsum_prior_offsetx_class, thr_wsum_prior_offsety_class, thr_max_weight;
	DOUBLE thr_wsum_sigma2_offset;
	MultidimArray<DOUBLE> thr_metadata, zeroArray;

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
	int number_valid_weight = 0 ;
	// exp_iclass loop does not always go from 0 to nr_classes!
	long int iorientclass_offset = exp_iclass * exp_nr_rot;
	size_t first_iorient = 0, last_iorient = 0;
	while (exp_iorient_ThreadTaskDistributor->getTasks(first_iorient, last_iorient))
	{
		for (long int iorient = first_iorient; iorient <= last_iorient; iorient++)
		{

			long int iorientclass = iorientclass_offset + iorient;

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

#ifdef TIMING
					// Only time one thread, as I also only time one MPI process
					if (thread_id == 0)
					{
						timer.tic(TIMING_WSUM_PROJ);
					}
#endif
					// Project the reference map (into Fref)
					if (!do_skip_maximization)
					{
						(mymodel.PPref[exp_iclass]).get2DFourierTransform(Fref, A, IS_INV);
					}

#ifdef TIMING
					// Only time one thread, as I also only time one MPI process
					if (thread_id == 0)
					{
						timer.toc(TIMING_WSUM_PROJ);
					}
#endif
					// Inside the loop over all translations and all part_id sum all shift Fimg's and their weights
					// Then outside this loop do the actual backprojection
					Fimg.initZeros(Fref);
					Fweight.initZeros(Fref);

					/// Now that reference projection has been made loop over someParticles!
					for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
					{
						// loop over all particles inside this ori_particle
						for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
						{
							long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
#ifdef DEBUG_CHECKSIZES
							if (ipart >= exp_starting_image_no.size())
							{
								std::cerr << "ipart= " << ipart << " starting_image_no.size()= " << exp_starting_image_no.size() << std::endl;
								REPORT_ERROR("ipart >= starting_image_no.size()");
							}
#endif
							// Which number was this image in the combined array of iseries and part_idpart_id
							long int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
							int group_id = mydata.getGroupId(part_id, exp_iseries);

#ifdef DEBUG_CHECKSIZES
							if (group_id >= mymodel.nr_groups)
							{
								std::cerr << "group_id= " << group_id << " ml_model.nr_groups= " << mymodel.nr_groups << std::endl;
								REPORT_ERROR("group_id >= ml_model.nr_groups");
							}
#endif

							if (!do_skip_maximization)
							{
								if (do_map)
								{
									Minvsigma2 = exp_local_Minvsigma2s[my_image_no];
								}
								// else Minvsigma2 was initialised to ones

								// Apply CTF to reference projection
								if (do_ctf_correction)
								{
									Mctf = exp_local_Fctfs[my_image_no];
									if (refs_are_ctf_corrected)
									{
										FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fref)
										{
											DIRECT_MULTIDIM_ELEM(Frefctf, n) = DIRECT_MULTIDIM_ELEM(Fref, n) * DIRECT_MULTIDIM_ELEM(Mctf, n);
										}
									}
									else
									{
										Frefctf = Fref;
									}
								}
								else
								{
									// initialise because there are multiple particles and Mctf gets selfMultiplied for scale_correction
									Mctf.initConstant(1.);
									Frefctf = Fref;
								}

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
									FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Frefctf)
									{
										DIRECT_MULTIDIM_ELEM(Frefctf, n) *= myscale;
									}
									// For CTF-terms in BP
									Mctf *= myscale;
								}
							} // end if !do_skip_maximization

							long int ihidden = iorientclass * exp_nr_trans;
							for (long int itrans = 0; itrans < exp_nr_trans; itrans++, ihidden++)
							{

								sampling.getTranslations(itrans, adaptive_oversampling, oversampled_translations);

								for (long int iover_trans = 0; iover_trans < exp_nr_oversampled_trans; iover_trans++)
								{

#ifdef DEBUG_CHECKSIZES
									if (iover_trans >= oversampled_translations.size())
									{
										std::cerr << "iover_trans= " << iover_trans << " oversampled_translations.size()= " << oversampled_translations.size() << std::endl;
										REPORT_ERROR("iover_trans >= oversampled_translations.size()");
									}
#endif

									// Only deal with this sampling point if its weight was significant
									long int ihidden_over = ihidden * exp_nr_oversampled_trans * exp_nr_oversampled_rot +
									                        iover_rot * exp_nr_oversampled_trans + iover_trans;

#ifdef DEBUG_CHECKSIZES
									if (ihidden_over >= XSIZE(exp_Mweight))
									{
										std::cerr << "ihidden_over= " << ihidden_over << " XSIZE(exp_Mweight)= " << XSIZE(exp_Mweight) << std::endl;
										REPORT_ERROR("ihidden_over >= XSIZE(exp_Mweight)");
									}
									if (ipart >= exp_significant_weight.size())
									{
										std::cerr << "ipart= " << ipart << " exp_significant_weight.size()= " << exp_significant_weight.size() << std::endl;
										REPORT_ERROR("ipart >= significant_weight.size()");
									}
									if (ipart >= exp_max_weight.size())
									{
										std::cerr << "ipart= " << ipart << " exp_max_weight.size()= " << exp_max_weight.size() << std::endl;
										REPORT_ERROR("ipart >= exp_max_weight.size()");
									}
									if (ipart >= exp_sum_weight.size())
									{
										std::cerr << "ipart= " << ipart << " exp_max_weight.size()= " << exp_sum_weight.size() << std::endl;
										REPORT_ERROR("ipart >= exp_sum_weight.size()");
									}
#endif
									DOUBLE weight = DIRECT_A2D_ELEM(exp_Mweight, ipart, ihidden_over);

									// Only sum weights for non-zero weights
									if (weight >= exp_significant_weight[ipart])
									{
#ifdef TIMING
										// Only time one thread, as I also only time one MPI process
										if (thread_id == 0)
										{
											timer.tic(TIMING_WSUM_DIFF2);
										}
#endif
										// Normalise the weight (do this after the comparison with exp_significant_weight!)
										weight /= exp_sum_weight[ipart];
										number_valid_weight++;
										if (!do_skip_maximization)
										{
											// Get the shifted image
											long int ishift = my_image_no * exp_nr_oversampled_trans * exp_nr_trans +
											                  itrans * exp_nr_oversampled_trans + iover_trans;
											Fimg_shift = exp_local_Fimgs_shifted[ishift];
											Fimg_shift_nomask = exp_local_Fimgs_shifted_nomask[ishift];

											// Store weighted sum of squared differences for sigma2_noise estimation
											FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Mresol_fine)
											{
												int ires = DIRECT_MULTIDIM_ELEM(Mresol_fine, n);
												if (ires > -1)
												{
													// Use FT of masked image for noise estimation!
													DOUBLE diff_real = (DIRECT_MULTIDIM_ELEM(Frefctf, n)).real - (DIRECT_MULTIDIM_ELEM(Fimg_shift, n)).real;
													DOUBLE diff_imag = (DIRECT_MULTIDIM_ELEM(Frefctf, n)).imag - (DIRECT_MULTIDIM_ELEM(Fimg_shift, n)).imag;
													DOUBLE wdiff2 = weight * (diff_real * diff_real + diff_imag * diff_imag);

													// group-wise sigma2_noise
													DIRECT_MULTIDIM_ELEM(thr_wsum_sigma2_noise[group_id], ires) += wdiff2;
													// For norm_correction
													thr_wsum_norm_correction[ipart] += wdiff2;
												}
											}

											// Store the weighted sums of the norm_correction terms
											if (do_scale_correction)
											{
												DOUBLE sumXA = 0.;
												DOUBLE sumA2 = 0.;
												FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Mresol_fine)
												{
													int ires = DIRECT_MULTIDIM_ELEM(Mresol_fine, n);
#ifdef DEBUG_CHECKSIZES
													if (ires >= XSIZE(thr_wsum_scale_correction_XA[ipart]))
													{
														std::cerr << "ires= " << ires << " XSIZE(thr_wsum_scale_correction_XA[ipart])= " << XSIZE(thr_wsum_scale_correction_XA[ipart]) << std::endl;
														REPORT_ERROR("ires >= XSIZE(thr_wsum_scale_correction_XA[ipart])");
													}
													if (ires >= XSIZE(thr_wsum_scale_correction_AA[ipart]))
													{
														std::cerr << "ires= " << ires << " XSIZE(thr_wsum_scale_correction_AA[ipart])= " << XSIZE(thr_wsum_scale_correction_AA[ipart]) << std::endl;
														REPORT_ERROR("ires >= XSIZE(thr_wsum_scale_correction_AA[ipart])");
													}
#endif

													// Once the reference becomes strongly regularised one does no longer want to store XA and AA!
													if (ires > -1 && DIRECT_A1D_ELEM(mymodel.data_vs_prior_class[exp_iclass], ires) > 3.)
													{
														sumXA = (DIRECT_MULTIDIM_ELEM(Frefctf, n)).real * (DIRECT_MULTIDIM_ELEM(Fimg_shift, n)).real;
														sumXA += (DIRECT_MULTIDIM_ELEM(Frefctf, n)).imag * (DIRECT_MULTIDIM_ELEM(Fimg_shift, n)).imag;
														DIRECT_A1D_ELEM(thr_wsum_scale_correction_XA[ipart], ires) += weight * sumXA;

														// This could be pre-calculated above...
														sumA2 = (DIRECT_MULTIDIM_ELEM(Frefctf, n)).real * (DIRECT_MULTIDIM_ELEM(Frefctf, n)).real;
														sumA2 += (DIRECT_MULTIDIM_ELEM(Frefctf, n)).imag * (DIRECT_MULTIDIM_ELEM(Frefctf, n)).imag;
														DIRECT_A1D_ELEM(thr_wsum_scale_correction_AA[ipart], ires) += weight * sumA2;
													}
												}
											}

											// Store sum of weights for this group
											thr_sumw_group[group_id] += weight;

											// Store weights for this class and orientation
											thr_wsum_pdf_class[exp_iclass] += weight;

											if (mymodel.ref_dim == 2)
											{
												// Also store weighted offset differences for prior_offsets of each class
												thr_wsum_prior_offsetx_class[exp_iclass] += weight * XX(exp_old_offset[my_image_no] + oversampled_translations[iover_trans]);
												thr_wsum_prior_offsety_class[exp_iclass] += weight * YY(exp_old_offset[my_image_no] + oversampled_translations[iover_trans]);

												// Store weighted sum2 of origin offsets (in Angstroms instead of pixels!!!)
												thr_wsum_sigma2_offset += weight * ((mymodel.prior_offset_class[exp_iclass] - exp_old_offset[my_image_no] - oversampled_translations[iover_trans]).sum2());

											}
											else
											{
												// Store weighted sum2 of origin offsets (in Angstroms instead of pixels!!!)
												thr_wsum_sigma2_offset += weight * ((exp_prior[my_image_no] - exp_old_offset[my_image_no] - oversampled_translations[iover_trans]).sum2());
											}

#ifdef DEBUG_CHECKSIZES
											if (idir >= XSIZE(thr_wsum_pdf_direction[exp_iclass]))
											{
												std::cerr << "idir= " << idir << " XSIZE(thr_wsum_pdf_direction[exp_iclass])= " << XSIZE(thr_wsum_pdf_direction[exp_iclass]) << std::endl;
												REPORT_ERROR("idir >= XSIZE(thr_wsum_pdf_direction[iclass])");
											}
#endif

											// Store weight for this direction of this class
											if (mymodel.orientational_prior_mode == NOPRIOR)
											{
												DIRECT_MULTIDIM_ELEM(thr_wsum_pdf_direction[exp_iclass], idir) += weight;
											}
											else
											{
												// In the case of orientational priors, get the original number of the direction back
												long int mydir = sampling.getDirectionNumberAlsoZeroPrior(idir);
												DIRECT_MULTIDIM_ELEM(thr_wsum_pdf_direction[exp_iclass], mydir) += weight;
											}

#ifdef TIMING
											// Only time one thread, as I also only time one MPI process
											if (thread_id == 0)
											{
												timer.toc(TIMING_WSUM_DIFF2);
											}
											// Only time one thread, as I also only time one MPI process
											if (thread_id == 0)
											{
												timer.tic(TIMING_WSUM_SUMSHIFT);
											}
#endif

											// Store sum of weight*SSNR*Fimg in data and sum of weight*SSNR in weight
											// Use the FT of the unmasked image to back-project in order to prevent reconstruction artefacts! SS 25oct11
											FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fimg_shift)
											{
												DOUBLE myctf = DIRECT_MULTIDIM_ELEM(Mctf, n);
												// Note that weightxinvsigma2 already contains the CTF!
												DOUBLE weightxinvsigma2 = weight * myctf * DIRECT_MULTIDIM_ELEM(Minvsigma2, n);
												// now Fimg stores sum of all shifted w*Fimg
												(DIRECT_MULTIDIM_ELEM(Fimg, n)).real += (DIRECT_MULTIDIM_ELEM(Fimg_shift_nomask, n)).real * weightxinvsigma2;
												(DIRECT_MULTIDIM_ELEM(Fimg, n)).imag += (DIRECT_MULTIDIM_ELEM(Fimg_shift_nomask, n)).imag * weightxinvsigma2;
												// now Fweight stores sum of all w
												// Note that CTF needs to be squared in Fweight, weightxinvsigma2 already contained one copy
												DIRECT_MULTIDIM_ELEM(Fweight, n) += weightxinvsigma2 * myctf;
											}

#ifdef TIMING
											// Only time one thread, as I also only time one MPI process
											if (thread_id == 0)
											{
												timer.toc(TIMING_WSUM_SUMSHIFT);
											}
#endif

										} // end if !do_skip_maximization

										// Keep track of max_weight and the corresponding optimal hidden variables
										if (weight > thr_max_weight[ipart])
										{
											// Store optimal image parameters
											thr_max_weight[ipart] = weight;

											// Calculate the angles back from the Euler matrix because for tilt series exp_R_mic may have changed them...
											//std::cerr << " ORI rot= " << rot << " tilt= " << tilt << " psi= " << psi << std::endl;
											Euler_matrix2angles(A.inv(), rot, tilt, psi);
											//std::cerr << " BACK rot= " << rot << " tilt= " << tilt << " psi= " << psi << std::endl;

											DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_ROT) = rot;
											DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_TILT) = tilt;
											DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_PSI) = psi;
											DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_XOFF) = XX(exp_old_offset[my_image_no]) + XX(oversampled_translations[iover_trans]);
											DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_YOFF) = YY(exp_old_offset[my_image_no]) + YY(oversampled_translations[iover_trans]);
											DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_CLASS) = (DOUBLE)exp_iclass + 1;
											DIRECT_A2D_ELEM(thr_metadata, my_image_no, METADATA_PMAX) = thr_max_weight[ipart];
										}

									} // end if weight >= exp_significant_weight
								} // end loop iover_trans
							} // end loop itrans
						}// end loop part_id (i)
					} // end loop ori_part_id
					if (!do_skip_maximization)
					{
#ifdef TIMING
						// Only time one thread, as I also only time one MPI process
						if (thread_id == 0)
						{
							timer.tic(TIMING_WSUM_BACKPROJ);
						}
#endif
						// Perform the actual back-projection.
						// This is done with the sum of all (in-plane) shifted Fimg's
						// Perform this inside a mutex
						global_mutex2.lock();
						(wsum_model.BPref[exp_iclass]).set2DFourierTransform(Fimg, A, IS_INV, &Fweight);
						global_mutex2.unlock();

#ifdef TIMING
						// Only time one thread, as I also only time one MPI process
						if (thread_id == 0)
						{
							timer.toc(TIMING_WSUM_BACKPROJ);
						}
#endif
					} // end if !do_skip_maximization

				}// end if iover_rot
			}// end loop do_proceed

		} // end loop ipsi
	} // end loop idir

	// Now, inside a global_mutex, update the weighted sums among all threads
	global_mutex.lock();
	std::cout << "The number of valid weight in CPU mode is: " << number_valid_weight << std::endl;

	if (!do_skip_maximization)
	{
		if (do_scale_correction)
		{
			for (int n = 0; n < exp_nr_particles; n++)
			{
				exp_wsum_scale_correction_XA[n] += thr_wsum_scale_correction_XA[n];
				exp_wsum_scale_correction_AA[n] += thr_wsum_scale_correction_AA[n];
			}
		}
		for (int n = 0; n < exp_nr_particles; n++)
		{
			exp_wsum_norm_correction[n] += thr_wsum_norm_correction[n];
		}
		for (int n = 0; n < mymodel.nr_groups; n++)
		{
			wsum_model.sigma2_noise[n] += thr_wsum_sigma2_noise[n];
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
	global_mutex.unlock();

	// Wait until all threads have finished
	global_barrier->wait();
#ifdef DEBUG_THREAD
	std::cerr << "leaving doThreadStoreWeightedSumsAllOrientations" << std::endl;
#endif


}

void MlOptimiser::storeWeightedSums()
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

	// In doThreadPrecalculateShiftedImagesCtfsAndInvSigma2s() the origin of the exp_local_Minvsigma2s was omitted.
	// Set those back here
	/*
	    for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	    {
	        // loop over all particles inside this ori_particle
	        for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
	        {
	            long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];

	            //std::cout << part_id << std::endl;

	            for (exp_iseries = 0; exp_iseries < mydata.getNrImagesInSeries(part_id); exp_iseries++)
	            {
	                // Re-get all shifted versions of the (current_sized) images, their (current_sized) CTFs and their inverted Sigma2 matrices
	                // This may be necessary for when using --strict_highres_exp. Otherwise norm estimation may become unstable!!
	                exp_ipart_ThreadTaskDistributor->reset(); // reset thread distribution tasks
	                global_ThreadManager->run(globalThreadPrecalculateShiftedImagesCtfsAndInvSigma2s);

	                int group_id = mydata.getGroupId(part_id, exp_iseries);
	                int my_image_no = exp_starting_image_no[ipart] + exp_iseries;
	                DIRECT_MULTIDIM_ELEM(exp_local_Minvsigma2s[my_image_no], 0) = 1. / (sigma2_fudge * DIRECT_A1D_ELEM(mymodel.sigma2_noise[group_id], 0));
	            }
	        }
	    }
	*/

	for (exp_iseries = 0; exp_iseries < mydata.getNrImagesInSeries((mydata.ori_particles[exp_my_first_ori_particle]).particles_id[0]); exp_iseries++)
	{
		exp_ipart_ThreadTaskDistributor->reset(); // reset thread distribution tasks
		//global_ThreadManager->run(globalThreadPrecalculateShiftedImagesCtfsAndInvSigma2s);
	}
	//for (exp_iseries = 0; exp_iseries < mydata.getNrImagesInSeries((mydata.ori_particles[exp_my_first_ori_particle]).particles_id[0]); exp_iseries++)

	for (long int ori_part_id = exp_my_first_ori_particle, ipart = 0; ori_part_id <= exp_my_last_ori_particle; ori_part_id++)
	{
		// loop over all particles inside this ori_particle
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];

			//std::cout << part_id << std::endl;

			for (exp_iseries = 0; exp_iseries < mydata.getNrImagesInSeries(part_id); exp_iseries++)
			{
				// Re-get all shifted versions of the (current_sized) images, their (current_sized) CTFs and their inverted Sigma2 matrices
				// This may be necessary for when using --strict_highres_exp. Otherwise norm estimation may become unstable!!
				//exp_ipart_ThreadTaskDistributor->reset(); // reset thread distribution tasks
				//global_ThreadManager->run(globalThreadPrecalculateShiftedImagesCtfsAndInvSigma2s);

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
		for (exp_iclass = iclass_min; exp_iclass <= iclass_max; exp_iclass++)
		{

			// The loops over all orientations are parallelised using threads

			//HERE!!
			exp_iorient_ThreadTaskDistributor->reset(); // reset thread distribution tasks
			global_ThreadManager->run(globalThreadStoreWeightedSumsAllOrientations);
		} // end loop iclass

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
					//TMP DEBUGGING
					/*
					if (!(iter == 1 && do_firstiter_cc) && DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM) > 10.)
					{
					    std::cerr << " mymodel.current_size= " << mymodel.current_size << " mymodel.ori_size= " << mymodel.ori_size << " part_id= " << part_id << std::endl;
					    std::cerr << " DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM)= " << DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM) << std::endl;
					    std::cerr << " mymodel.avg_norm_correction= " << mymodel.avg_norm_correction << std::endl;
					    std::cerr << " exp_wsum_norm_correction[ipart]= " << exp_wsum_norm_correction[ipart] << std::endl;
					    std::cerr << " old_norm_correction= " << old_norm_correction << std::endl;
					    std::cerr << " wsum_model.avg_norm_correction= " << wsum_model.avg_norm_correction << std::endl;
					    std::cerr << " group_id= " << group_id << " mymodel.scale_correction[group_id]= " << mymodel.scale_correction[group_id] << std::endl;
					    std::cerr << " mymodel.sigma2_noise[group_id]= " << mymodel.sigma2_noise[group_id] << std::endl;
					    std::cerr << " wsum_model.sigma2_noise[group_id]= " << wsum_model.sigma2_noise[group_id] << std::endl;
					    std::cerr << " exp_power_imgs[my_image_no]= " << exp_power_imgs[my_image_no] << std::endl;
					    std::cerr << " exp_wsum_scale_correction_XA[ipart]= " << exp_wsum_scale_correction_XA[ipart] << " exp_wsum_scale_correction_AA[ipart]= " << exp_wsum_scale_correction_AA[ipart] << std::endl;
					    std::cerr << " wsum_model.wsum_signal_product_spectra[group_id]= " << wsum_model.wsum_signal_product_spectra[group_id] << " wsum_model.wsum_reference_power_spectra[group_id]= " << wsum_model.wsum_reference_power_spectra[group_id] << std::endl;
					    std::cerr << " exp_min_diff2[ipart]= " << exp_min_diff2[ipart] << std::endl;
					    std::cerr << " ml_model.scale_correction[group_id]= " << mymodel.scale_correction[group_id] << std::endl;
					    std::cerr << " exp_significant_weight[ipart]= " << exp_significant_weight[ipart] << std::endl;
					    std::cerr << " exp_max_weight[ipart]= " << exp_max_weight[ipart] << std::endl;
					    mymodel.write("debug");
					    std::cerr << "written debug_model.star" << std::endl;
					    REPORT_ERROR("MlOptimiser::storeWeightedSums ERROR: normalization is larger than 10");
					}
					*/

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

/** Monitor the changes in the optimal translations, orientations and class assignments for some particles */
void MlOptimiser::monitorHiddenVariableChanges(long int my_first_ori_particle, long int my_last_ori_particle)
{

	for (long int ori_part_id = my_first_ori_particle, my_image_no = 0; ori_part_id <= my_last_ori_particle; ori_part_id++)
	{

#ifdef DEBUG_CHECKSIZES
		if (ori_part_id >= mydata.ori_particles.size())
		{
			std::cerr << "ori_part_id= " << ori_part_id << " mydata.ori_particles.size()= " << mydata.ori_particles.size() << std::endl;
			REPORT_ERROR("ori_part_id >= mydata.ori_particles.size()");
		}
#endif

		// loop over all particles inside this ori_particle
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++, my_image_no++)
			{
				long int img_id = mydata.getImageId(part_id, iseries);

#ifdef DEBUG_CHECKSIZES
				if (img_id >= mydata.MDimg.numberOfObjects())
				{
					std::cerr << "img_id= " << img_id << " mydata.MDimg.numberOfObjects()= " << mydata.MDimg.numberOfObjects() << std::endl;
					REPORT_ERROR("img_id >= mydata.MDimg.numberOfObjects()");
				}
				if (my_image_no >= YSIZE(exp_metadata))
				{
					std::cerr << "my_image_no= " << my_image_no << " YSIZE(exp_metadata)= " << YSIZE(exp_metadata) << std::endl;
					REPORT_ERROR("my_image_no >= YSIZE(exp_metadata)");
				}
#endif

				// Old optimal parameters
				DOUBLE old_rot, old_tilt, old_psi, old_xoff, old_yoff, old_zoff = 0.;
				int old_iclass;
				mydata.MDimg.getValue(EMDL_ORIENT_ROT,  old_rot, img_id);
				mydata.MDimg.getValue(EMDL_ORIENT_TILT, old_tilt, img_id);
				mydata.MDimg.getValue(EMDL_ORIENT_PSI,  old_psi, img_id);
				mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X, old_xoff, img_id);
				mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y, old_yoff, img_id);
				mydata.MDimg.getValue(EMDL_PARTICLE_CLASS, old_iclass, img_id);

				// New optimal parameters
				DOUBLE rot = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT);
				DOUBLE tilt = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT);
				DOUBLE psi = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI);
				DOUBLE xoff = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF);
				DOUBLE yoff = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF);
				DOUBLE zoff = 0.;
				int iclass = (int)DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CLASS);

				// Some orientational distance....
				sum_changes_optimal_orientations += sampling.calculateAngularDistance(rot, tilt, psi, old_rot, old_tilt, old_psi);
				sum_changes_optimal_offsets += (xoff - old_xoff) * (xoff - old_xoff) + (yoff - old_yoff) * (yoff - old_yoff) + (zoff - old_zoff) * (zoff - old_zoff);
				if (iclass != old_iclass)
				{
					sum_changes_optimal_classes += 1.;
				}
				sum_changes_count += 1.;
			} // end loop iseries
		} // end loop part_id (i)
	} //end loop ori_part_id


}

void MlOptimiser::updateOverallChangesInHiddenVariables()
{

	// Calculate hidden variable changes
	current_changes_optimal_classes = sum_changes_optimal_classes / sum_changes_count;
	current_changes_optimal_orientations = sum_changes_optimal_orientations / sum_changes_count;
	current_changes_optimal_offsets = sqrt(sum_changes_optimal_offsets / (2. * sum_changes_count));

	// Reset the sums
	sum_changes_optimal_classes = 0.;
	sum_changes_optimal_orientations = 0.;
	sum_changes_optimal_offsets = 0.;
	sum_changes_count = 0.;

	// Update nr_iter_wo_large_hidden_variable_changes if all three assignment types are within 3% of the smallest thus far
	if (1.03 * current_changes_optimal_classes >= smallest_changes_optimal_classes &&
	        1.03 * current_changes_optimal_offsets >= smallest_changes_optimal_offsets &&
	        1.03 * current_changes_optimal_orientations >= smallest_changes_optimal_orientations)
	{
		nr_iter_wo_large_hidden_variable_changes++;
	}
	else
	{
		nr_iter_wo_large_hidden_variable_changes = 0;
	}

	// Update smallest changes in hidden variables thus far
	if (current_changes_optimal_classes < smallest_changes_optimal_classes)
	{
		smallest_changes_optimal_classes = ROUND(current_changes_optimal_classes);
	}
	if (current_changes_optimal_offsets < smallest_changes_optimal_offsets)
	{
		smallest_changes_optimal_offsets = current_changes_optimal_offsets;
	}
	if (current_changes_optimal_orientations < smallest_changes_optimal_orientations)
	{
		smallest_changes_optimal_orientations = current_changes_optimal_orientations;
	}


}


void MlOptimiser::calculateExpectedAngularErrors(long int my_first_ori_particle, long int my_last_ori_particle)
{

	long int n_trials = 0;
	exp_starting_image_no.clear();
	exp_nr_images = 0;
	for (long int ori_part_id = my_first_ori_particle, my_image_no = 0, ipart = 0; ori_part_id <= my_last_ori_particle; ori_part_id++)
	{
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			exp_starting_image_no.push_back(exp_nr_images);
			exp_nr_images += mydata.getNrImagesInSeries(part_id);
			n_trials++;
		}
	}

	// Set exp_current_image_size to the coarse_size to calculate exepcted angular errors
	if (strict_highres_exp > 0. && !do_acc_currentsize_despite_highres_exp)
	{
		// Use smaller images in both passes and keep a maximum on coarse_size, just like in FREALIGN
		exp_current_image_size = coarse_size;
	}
	else
	{
		// Use smaller images in the first pass, but larger ones in the second pass
		exp_current_image_size = mymodel.current_size;
	}

	// Separate angular error estimate for each of the classes
	acc_rot = acc_trans = 999.; // later XMIPP_MIN will be taken to find the best class...

	// P(X | X_1) / P(X | X_2) = exp ( |F_1 - F_2|^2 / (-2 sigma2) )
	// exp(-4.60517) = 0.01
	DOUBLE pvalue = 4.60517;
	//std::cout << "value of nrtrials : " <<  n_trials << "  " << mymodel.nr_classes << "  " << mymodel.ref_dim << std::endl;
	//std::cout << "iclass pdf " << mymodel.pdf_class[0] << " " <<  mymodel.pdf_class[1]  << " " << mymodel.pdf_class[2] <<  " " << mymodel.pdf_class[3] << std::endl;
	std::cout << " Estimating accuracies in the orientational assignment ... " << std::endl;
	init_progress_bar(n_trials * mymodel.nr_classes);
	for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
	{

		// Don't do this for (almost) empty classes
		if (mymodel.pdf_class[iclass] < 0.01)
		{
			mymodel.acc_rot[iclass]   = 999.;
			mymodel.acc_trans[iclass] = 999.;
			continue;
		}
		//std::cout << "iclass pdf" << mymodel.pdf_class[iclass] << std::endl;
		// Initialise the orientability arrays that will be written out in the model.star file
		// These are for the user's information only: nothing will be actually done with them
#ifdef DEBUG_CHECKSIZES
		if (iclass >= (mymodel.orientability_contrib).size())
		{
			std::cerr << "iclass= " << iclass << " (mymodel.orientability_contrib).size()= " << (mymodel.orientability_contrib).size() << std::endl;
			REPORT_ERROR("iclass >= (mymodel.orientability_contrib).size()");
		}
#endif
		(mymodel.orientability_contrib)[iclass].initZeros(mymodel.ori_size / 2 + 1);

		DOUBLE acc_rot_class = 0.;
		DOUBLE acc_trans_class = 0.;
		// Particles are already in random order, so just move from 0 to n_trials
		for (long int ori_part_id = my_first_ori_particle, my_image_no = 0, ipart = 0; ori_part_id <= my_last_ori_particle; ori_part_id++)
		{
			for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++)
			{
				long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];

				// Search 2 times: ang and off
				// Don't estimate rotational accuracies if we're doing do_skip_rotate (for faster movie-frame alignment)
				int imode_start = (do_skip_rotate) ? 1 : 0;
				for (int imode = imode_start; imode < 2; imode++)
				{
					DOUBLE ang_error = 0.;
					DOUBLE sh_error = 0.;
					DOUBLE ang_step;
					DOUBLE sh_step;
					DOUBLE my_snr = 0.;

					// Search for ang_error and sh_error where there are at least 3-sigma differences!
					// 13feb12: change for explicit probability at P=0.01
					while (my_snr <= pvalue)
					{
						// Graduallly increase the step size
						if (ang_error < 0.2)
						{
							ang_step = 0.05;
						}
						else if (ang_error < 1.)
						{
							ang_step = 0.1;
						}
						else if (ang_error < 2.)
						{
							ang_step = 0.2;
						}
						else if (ang_error < 5.)
						{
							ang_step = 0.5;
						}
						else if (ang_error < 10.)
						{
							ang_step = 1.0;
						}
						else if (ang_error < 20.)
						{
							ang_step = 2;
						}
						else
						{
							ang_step = 5.0;
						}

						if (sh_error < 0.2)
						{
							sh_step = 0.05;
						}
						else if (sh_error < 1.)
						{
							sh_step = 0.1;
						}
						else if (sh_error < 2.)
						{
							sh_step = 0.2;
						}
						else if (sh_error < 5.)
						{
							sh_step = 0.5;
						}
						else if (sh_error < 10.)
						{
							sh_step = 1.0;
						}
						else
						{
							sh_step = 2.0;
						}

						ang_error += ang_step;
						sh_error += sh_step;

						// Prevent an endless while by putting boundaries on ang_error and sh_error
						if ((imode == 0 && ang_error > 30.) || (imode == 1 && sh_error > 10.))
						{
							break;
						}

						init_random_generator(random_seed + part_id);

						// Loop over all images in the series
						// TODO: check this for series!!
						// Initialise the my_snr value (accumulate its sum for all images in the series!!)
						my_snr = 0.;
						for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++)
						{

							int my_image_no = exp_starting_image_no.at(ipart) + iseries;

							Matrix2D<DOUBLE> R_mic(3, 3);
							R_mic(0, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_0);
							R_mic(0, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_1);
							R_mic(0, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_2);
							R_mic(1, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_0);
							R_mic(1, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_1);
							R_mic(1, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_2);
							R_mic(2, 0) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_0);
							R_mic(2, 1) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_1);
							R_mic(2, 2) = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_2);

							int group_id = mydata.getGroupId(part_id, iseries);
#ifdef DEBUG_CHECKSIZES
							if (group_id  >= mymodel.sigma2_noise.size())
							{
								std::cerr << "group_id = " << group_id << " mymodel.sigma2_noise.size()= " << mymodel.sigma2_noise.size() << std::endl;
								REPORT_ERROR("group_id  >= mymodel.sigma2_noise.size()");
							}
#endif
							MultidimArray<Complex > F1, F2;
							MultidimArray<DOUBLE> Fctf;
							Matrix2D<DOUBLE> A1, A2;


							// TODO: get values through exp_metadata?!
							DOUBLE rot1 = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT);
							DOUBLE tilt1 = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT);
							DOUBLE psi1 = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI);
							DOUBLE xoff1 = 0.;
							DOUBLE yoff1 = 0.;

							// Get the FT of the first image
							F1.initZeros(exp_current_image_size, exp_current_image_size / 2 + 1);

							Euler_angles2matrix(rot1, tilt1, psi1, A1);
							A1 = R_mic * A1.inv();
							(mymodel.PPref[iclass]).get2DFourierTransform(F1, A1, IS_INV);

							// Apply the angular or shift error
							DOUBLE rot2 = rot1;
							DOUBLE tilt2 = tilt1;
							DOUBLE psi2 = psi1;
							Matrix1D<DOUBLE> shift(2);
							XX(shift) = xoff1;
							YY(shift) = yoff1;
							// Perturb psi or xoff , depending on the mode
							if (imode == 0)
							{
								if (mymodel.ref_dim == 3)
								{
									// Randomly change rot, tilt or psi
									DOUBLE ran = rnd_unif();
									if (ran < 0.3333)
									{
										rot2 = rot1 + ang_error;
									}
									else if (ran < 0.6667)
									{
										tilt2 = tilt1 + ang_error;
									}
									else
									{
										psi2  = psi1 + ang_error;
									}
								}
								else
								{
									psi2  = psi1 + ang_error;
								}
							}
							else
							{
								// Randomly change xoff or yoff
								DOUBLE ran = rnd_unif();
								if (ran < 0.5)
								{
									XX(shift) = xoff1 + sh_error;
								}
								else
								{
									YY(shift) = yoff1 + sh_error;
								}
							}
							// Get the FT of the second image
							F2.initZeros(exp_current_image_size, exp_current_image_size / 2 + 1);
							Euler_angles2matrix(rot2, tilt2, psi2, A2);
							A2 = R_mic * A2.inv();
							(mymodel.PPref[iclass]).get2DFourierTransform(F2, A2, IS_INV);
							if (ABS(XX(shift)) > 0. || ABS(YY(shift)) > 0.)
								// shiftImageInFourierTransform takes shifts in pixels!
							{
								shiftImageInFourierTransform(F2, F2, (DOUBLE) mymodel.ori_size, -shift);
							}
							// Apply CTF to F1 and F2 if necessary
							if (do_ctf_correction)
							{
								CTF ctf;

								ctf.setValues(DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_DEFOCUS_U),
								              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_DEFOCUS_V),
								              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_DEFOCUS_ANGLE),
								              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_VOLTAGE),
								              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_CS),
								              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_Q0),
								              DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_BFAC));

								Fctf.resize(F1);
								ctf.getFftwImage(Fctf, mymodel.ori_size, mymodel.ori_size, mymodel.pixel_size, ctf_phase_flipped, only_flip_phases, intact_ctf_first_peak, true);
#ifdef DEBUG_CHECKSIZES
								if (!Fctf.sameShape(F1) || !Fctf.sameShape(F2))
								{
									std::cerr << " Fctf: ";
									Fctf.printShape(std::cerr);
									std::cerr << " F1:   ";
									F1.printShape(std::cerr);
									std::cerr << " F2:   ";
									F2.printShape(std::cerr);
									REPORT_ERROR("ERROR: Fctf has a different shape from F1 and F2");
								}
#endif
								//FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F1)
								//{
								//  std::cout << " the value of my_snr " << my_snr << "  " <<  DIRECT_MULTIDIM_ELEM(F1, n).real << "  " << DIRECT_MULTIDIM_ELEM(F2, n).real <<  "  "  << DIRECT_MULTIDIM_ELEM(Fctf, n)<< "  "  << std::endl;
								//}

								FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F1)
								{
									DIRECT_MULTIDIM_ELEM(F1, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
									DIRECT_MULTIDIM_ELEM(F2, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
								}
							}

							MultidimArray<int>* myMresol = (YSIZE(F1) == coarse_size) ? &Mresol_coarse : &Mresol_fine;
							FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F1)
							{
								int ires = DIRECT_MULTIDIM_ELEM(*myMresol, n);
								if (ires > 0)
								{
									my_snr += norm(DIRECT_MULTIDIM_ELEM(F1, n) - DIRECT_MULTIDIM_ELEM(F2, n)) / (2 * sigma2_fudge * mymodel.sigma2_noise[group_id](ires));
								}
								//std::cout << " the value of my_snr " << my_snr << "  " <<  DIRECT_MULTIDIM_ELEM(F1, n).real << "  " << DIRECT_MULTIDIM_ELEM(F2, n).real <<  "  "  << DIRECT_MULTIDIM_ELEM(Fctf, n)<< "  "  << std::endl;

							}
							//std::cout << " the value of my_snr " << my_snr << "  " <<  YSIZE(F1) << "  " << coarse_size <<  "  "  << (*myMresol).zyxdim << "  " << pvalue << std::endl;
							// Only for the psi-angle and the translations, and only when my_prob < 0.01 calculate a histogram of the contributions at each resolution shell
							if (my_snr > pvalue && imode == 0)
							{
								FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F1)
								{
									int ires = DIRECT_MULTIDIM_ELEM(*myMresol, n);
									if (ires > 0)
										mymodel.orientability_contrib[iclass](ires) +=
										    norm(DIRECT_MULTIDIM_ELEM(F1, n) - DIRECT_MULTIDIM_ELEM(F2, n)) / ((2 * sigma2_fudge * mymodel.sigma2_noise[group_id](ires)));
								}

							}

						} // end for iseries

					} // end while my_snr >= pvalue
					if (imode == 0)
					{
						acc_rot_class += ang_error;
					}
					else if (imode == 1)
					{
						acc_trans_class += sh_error;
					}
				} // end for imode

			}// end for part_id

			progress_bar(n_trials * iclass + ipart);
		} // end for ori_part_id

		mymodel.acc_rot[iclass]   = acc_rot_class / (DOUBLE)n_trials;
		mymodel.acc_trans[iclass] = acc_trans_class / (DOUBLE)n_trials;

		// Store normalised spectral contributions to orientability
		if (mymodel.orientability_contrib[iclass].sum() > 0.)
		{
			mymodel.orientability_contrib[iclass]   /= mymodel.orientability_contrib[iclass].sum();
		}

		// Keep the orientational accuracy of the best class for the auto-sampling approach
		acc_rot     = XMIPP_MIN(mymodel.acc_rot[iclass], acc_rot);
		acc_trans   = XMIPP_MIN(mymodel.acc_trans[iclass], acc_trans);


		// Richard's formula with Greg's constant
		//DOUBLE b_orient = (acc_rot_class*acc_rot_class* particle_diameter*particle_diameter) / 3000.;
		//std::cout << " + expected B-factor from the orientational errors = "
		//      << b_orient<<std::endl;
		// B=8 PI^2 U^2
		//std::cout << " + expected B-factor from the translational errors = "
		//      << 8 * PI * PI * mymodel.pixel_size * mymodel.pixel_size * acc_trans_class * acc_trans_class << std::endl;

	} // end loop iclass
	progress_bar(n_trials * mymodel.nr_classes);


	std::cout << " Auto-refine: Estimated accuracy angles= " << acc_rot << " degrees; offsets= " << acc_trans << " pixels" << std::endl;
	// Warn for inflated resolution estimates
	if (acc_rot > 10.)
	{
		std::cout << " Auto-refine: WARNING: The angular accuracy is worse than 10 degrees, so basically you cannot align your particles (yet)!" << std::endl;
		std::cout << " Auto-refine: WARNING: You probably need not worry if the accuracy improves during the next few iterations." << std::endl;
		std::cout << " Auto-refine: WARNING: However, if the problem persists it may lead to spurious FSC curves, so be wary of inflated resolution estimates..." << std::endl;
		std::cout << " Auto-refine: WARNING: Sometimes it is better to tune resolution yourself by adjusting T in a 3D-classification with a single class." << std::endl;
	}

}

void MlOptimiser::updateAngularSampling(bool verb)
{

	if (!do_split_random_halves)
	{
		REPORT_ERROR("MlOptimiser::updateAngularSampling: BUG! updating of angular sampling should only happen for gold-standard (auto-) refinements.");
	}

	if (do_realign_movies)
	{

		// A. Adjust translational sampling to 75% of estimated accuracy
		DOUBLE new_step = XMIPP_MIN(1.5, 0.75 * acc_trans) * std::pow(2., adaptive_oversampling);

		// Search ranges are three times the estimates std.dev. in the offsets
		DOUBLE new_range = 3. * sqrt(mymodel.sigma2_offset);

		// Prevent too narrow searches: always at least 3x3 pixels in the coarse search
		if (new_range < 1.5 * new_step)
		{
			new_range = 1.5 * new_step;
		}

		// Also prevent too wide searches: that will lead to memory problems:
		// Just use coarser step size and hope things will settle down later...
		if (new_range > 4. * new_step)
		{
			new_step = new_range / 4.;
		}

		sampling.setTranslations(new_step, new_range);

		if (!do_skip_rotate)
		{
			// B. Find the healpix order that corresponds to at least 50% of the estimated rotational accuracy
			DOUBLE angle_range = sqrt(mymodel.sigma2_rot) * 3.;
			DOUBLE new_ang_step, new_ang_step_wo_over;
			int new_hp_order;
			for (new_hp_order = 0; new_hp_order < 8; new_hp_order++)
			{

				new_ang_step = 360. / (6 * ROUND(std::pow(2., new_hp_order + adaptive_oversampling)));
				new_ang_step_wo_over = 2. * new_ang_step;
				// Only consider healpix orders that gives at least more than one (non-oversampled) samplings within the local angular searches
				if (new_ang_step_wo_over > angle_range)
				{
					continue;
				}
				// If sampling is at least twice as fine as the estimated rotational accuracy, then use this sampling
				if (new_ang_step < 0.50 * acc_rot)
				{
					break;
				}
			}

			if (new_hp_order != sampling.healpix_order)
			{
				// Set the new sampling in the sampling-object
				sampling.setOrientations(new_hp_order, new_ang_step * std::pow(2., adaptive_oversampling));
				// Resize the pdf_direction arrays to the correct size and fill with an even distribution
				mymodel.initialisePdfDirection(sampling.NrDirections(0, true));
				// Also reset the nr_directions in wsum_model
				wsum_model.nr_directions = mymodel.nr_directions;
				// Also resize and initialise wsum_model.pdf_direction for each class!
				for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
				{
					wsum_model.pdf_direction[iclass].initZeros(mymodel.nr_directions);
				}
			}
		}
	}
	else
	{

		if (do_skip_rotate)
		{
			REPORT_ERROR("ERROR: --skip_rotate can only be used in classification or in movie-frame refinement ...");
		}

		// Only change the sampling if the resolution has not improved during the last 2 iterations
		// AND the hidden variables have not changed during the last 2 iterations
		DOUBLE old_rottilt_step = sampling.getAngularSampling(adaptive_oversampling);

		// Only use a finer angular sampling is the angular accuracy is still above 75% of the estimated accuracy
		// If it is already below, nothing will change and eventually nr_iter_wo_resol_gain or nr_iter_wo_large_hidden_variable_changes will go above MAX_NR_ITER_WO_RESOL_GAIN
		if (nr_iter_wo_resol_gain >= MAX_NR_ITER_WO_RESOL_GAIN && nr_iter_wo_large_hidden_variable_changes >= MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES)
		{
			// Old rottilt step is already below 75% of estimated accuracy: have to stop refinement
			if (old_rottilt_step < 0.75 * acc_rot)
			{
				// don't change angular sampling, as it is already fine enough
				has_fine_enough_angular_sampling = true;

			}
			else
			{
				has_fine_enough_angular_sampling = false;

				// A. Use translational sampling as suggested by acc_trans

				// Prevent very coarse translational samplings: max 1.5
				// Also stay a bit on the safe side with the translational sampling: 75% of estimated accuracy
				DOUBLE new_step = XMIPP_MIN(1.5, 0.75 * acc_trans) * std::pow(2., adaptive_oversampling);
				// Search ranges are five times the last observed changes in offsets
				DOUBLE new_range = 5. * current_changes_optimal_offsets;
				// New range can only become 30% bigger than the previous range (to prevent very slow iterations in the beginning)
				new_range = XMIPP_MIN(1.3 * sampling.offset_range, new_range);
				// Prevent too narrow searches: always at least 3x3 pixels in the coarse search
				if (new_range < 1.5 * new_step)
				{
					new_range = 1.5 * new_step;
				}
				// Also prevent too wide searches: that will lead to memory problems:
				// If steps size < 1/4th of search range, then decrease search range by 50%
				if (new_range > 4. * new_step)
				{
					new_range /= 2.;
				}
				//If even that was not enough: use coarser step size and hope things will settle down later...
				if (new_range > 4. * new_step)
				{
					new_step = new_range / 4.;
				}
				sampling.setTranslations(new_step, new_range);

				// B. Use twice as fine angular sampling
				int new_hp_order;
				DOUBLE new_rottilt_step, new_psi_step;
				if (mymodel.ref_dim == 3)
				{
					new_hp_order = sampling.healpix_order + 1;
					new_rottilt_step = new_psi_step = 360. / (6 * ROUND(std::pow(2., new_hp_order + adaptive_oversampling)));
				}
				else if (mymodel.ref_dim == 2)
				{
					new_hp_order = sampling.healpix_order;
					new_psi_step = sampling.getAngularSampling() / 2.;
				}
				else
				{
					REPORT_ERROR("MlOptimiser::autoAdjustAngularSampling BUG: ref_dim should be two or three");
				}

				// Set the new sampling in the sampling-object
				sampling.setOrientations(new_hp_order, new_psi_step * std::pow(2., adaptive_oversampling));

				// Resize the pdf_direction arrays to the correct size and fill with an even distribution
				mymodel.initialisePdfDirection(sampling.NrDirections(0, true));

				// Also reset the nr_directions in wsum_model
				wsum_model.nr_directions = mymodel.nr_directions;

				// Also resize and initialise wsum_model.pdf_direction for each class!
				for (int iclass = 0; iclass < mymodel.nr_classes; iclass++)
				{
					wsum_model.pdf_direction[iclass].initZeros(mymodel.nr_directions);
				}

				// Reset iteration counters
				nr_iter_wo_resol_gain = 0;
				nr_iter_wo_large_hidden_variable_changes = 0;

				// Reset smallest changes hidden variables
				smallest_changes_optimal_classes = 9999999;
				smallest_changes_optimal_offsets = 999.;
				smallest_changes_optimal_orientations = 999.;

				// If the angular sampling is smaller than autosampling_hporder_local_searches, then use local searches of +/- 6 times the angular sampling
				if (new_hp_order >= autosampling_hporder_local_searches)
				{
					// Switch ON local angular searches
					mymodel.orientational_prior_mode = PRIOR_ROTTILT_PSI;
					sampling.orientational_prior_mode = PRIOR_ROTTILT_PSI;
					mymodel.sigma2_rot = mymodel.sigma2_tilt = mymodel.sigma2_psi = 2. * 2. * new_rottilt_step * new_rottilt_step;
					nr_pool = max_nr_pool = 1;
				}

			}

		}
	}

	// Print to screen
	if (verb)
	{
		std::cout << " Auto-refine: Angular step= " << sampling.getAngularSampling(adaptive_oversampling) << " degrees; local searches= ";
		if (sampling.orientational_prior_mode == NOPRIOR)
		{
			std:: cout << "false" << std::endl;
		}
		else
		{
			std:: cout << "true" << std::endl;
		}
		std::cout << " Auto-refine: Offset search range= " << sampling.offset_range << " pixels; offset step= " << sampling.getTranslationalSampling(adaptive_oversampling) << " pixels" << std::endl;
	}

}

void MlOptimiser::checkConvergence()
{

	if (do_realign_movies)
	{
		// only resolution needs to be stuck
		// Since there does not seem to be any improvement (and sometimes even the opposite)
		// of performing more than one iteration with the movie frames, just perform a single iteration
		//if (nr_iter_wo_resol_gain >= MAX_NR_ITER_WO_RESOL_GAIN)
		//{
		//  has_converged = true;
		//  do_join_random_halves = true;
		//  // movies were already use all data until Nyquist
		//}
	}
	else
	{
		has_converged = false;
		if(!extra_iter){
		if (has_fine_enough_angular_sampling && nr_iter_wo_resol_gain >= MAX_NR_ITER_WO_RESOL_GAIN && nr_iter_wo_large_hidden_variable_changes >= MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES)
		{
			has_converged = true;
			do_join_random_halves = true;
			// In the last iteration, include all data until Nyquist
			do_use_all_data = true;
		}}
		/*else{
		if(iter!=nr_iter)
		{
			has_converged = false;
                        do_join_random_halves = false;
                        // In the last iteration, include all data until Nyquist
                        do_use_all_data = false;
		}
		else
		{
			has_converged = true;
                        do_join_random_halves = true;
                        // In the last iteration, include all data until Nyquist
                        do_use_all_data = true;
		}
		}*/
	}

}

void MlOptimiser::printConvergenceStats()
{

	std::cout << " Auto-refine: Iteration= " << iter << std::endl;
	std::cout << " Auto-refine: Resolution= " << 1. / mymodel.current_resolution << " (no gain for " << nr_iter_wo_resol_gain << " iter) " << std::endl;
	std::cout << " Auto-refine: Changes in angles= " << current_changes_optimal_orientations << " degrees; and in offsets= " << current_changes_optimal_offsets
	          << " pixels (no gain for " << nr_iter_wo_large_hidden_variable_changes << " iter) " << std::endl;

	if (has_converged)
	{
		std::cout << " Auto-refine: Refinement has converged, entering last iteration where two halves will be combined..." << std::endl;
		if (!do_realign_movies)
		{
			std::cout << " Auto-refine: The last iteration will use data to Nyquist frequency, which may take more CPU and RAM." << std::endl;
		}
	}

}

void MlOptimiser::setMetaDataSubset(int first_ori_particle_id, int last_ori_particle_id)
{

	for (long int ori_part_id = first_ori_particle_id, my_image_no = 0; ori_part_id <= last_ori_particle_id; ori_part_id++)
	{

#ifdef DEBUG_CHECKSIZES
		if (ori_part_id >= mydata.ori_particles.size())
		{
			std::cerr << "ori_part_id= " << ori_part_id << " mydata.ori_particles.size()= " << mydata.ori_particles.size() << std::endl;
			REPORT_ERROR("ori_part_id >= mydata.ori_particles.size()");
		}
#endif

		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];

			for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++, my_image_no++)
			{

				long int img_id = mydata.getImageId(part_id, iseries);

#ifdef DEBUG_CHECKSIZES
				if (img_id >= mydata.MDimg.numberOfObjects())
				{
					std::cerr << "img_id= " << img_id << " mydata.MDimg.numberOfObjects()= " << mydata.MDimg.numberOfObjects() << std::endl;
					REPORT_ERROR("img_id >= mydata.MDimg.numberOfObjects()");
				}
				if (my_image_no >= YSIZE(exp_metadata))
				{
					std::cerr << "my_image_no= " << my_image_no << " YSIZE(exp_metadata)= " << YSIZE(exp_metadata) << std::endl;
					REPORT_ERROR("my_image_no >= YSIZE(exp_metadata)");
				}
#endif
				mydata.MDimg.setValue(EMDL_ORIENT_ROT,  DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT), img_id);
				mydata.MDimg.setValue(EMDL_ORIENT_TILT, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT), img_id);
				mydata.MDimg.setValue(EMDL_ORIENT_PSI,  DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI), img_id);
				mydata.MDimg.setValue(EMDL_ORIENT_ORIGIN_X, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF), img_id);
				mydata.MDimg.setValue(EMDL_ORIENT_ORIGIN_Y, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF), img_id);
				mydata.MDimg.setValue(EMDL_PARTICLE_CLASS, (int)DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CLASS) , img_id);
				mydata.MDimg.setValue(EMDL_PARTICLE_DLL,  DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_DLL), img_id);
				mydata.MDimg.setValue(EMDL_PARTICLE_PMAX, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PMAX), img_id);
				mydata.MDimg.setValue(EMDL_PARTICLE_NR_SIGNIFICANT_SAMPLES, (int)DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NR_SIGN), img_id);
				mydata.MDimg.setValue(EMDL_IMAGE_NORM_CORRECTION, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM), img_id);
				
				// For the moment, CTF, prior and transformation matrix info is NOT updated...
				DOUBLE prior_x = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF_PRIOR);
				DOUBLE prior_y = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF_PRIOR);
				if (prior_x < 999.)
				{
					mydata.MDimg.setValue(EMDL_ORIENT_ORIGIN_X_PRIOR, prior_x, img_id);
				}
				if (prior_y < 999.)
				{
					mydata.MDimg.setValue(EMDL_ORIENT_ORIGIN_Y_PRIOR, prior_y, img_id);
				}
			}
		}
	}

}

void MlOptimiser::getMetaAndImageDataSubset(int first_ori_particle_id, int last_ori_particle_id, bool do_also_imagedata)
{

	int nr_images = 0;
	for (long int ori_part_id = first_ori_particle_id; ori_part_id <= last_ori_particle_id; ori_part_id++)
	{

#ifdef DEBUG_CHECKSIZES
		if (ori_part_id >= mydata.ori_particles.size())
		{
			std::cerr << "ori_part_id= " << ori_part_id << " mydata.ori_particles.size()= " << mydata.ori_particles.size() << std::endl;
			REPORT_ERROR("ori_part_id >= mydata.ori_particles.size()");
		}
#endif

		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			nr_images += mydata.getNrImagesInSeries(part_id);
		}
	}

	exp_metadata.initZeros(nr_images, METADATA_LINE_LENGTH);
	if (has_converged && do_use_reconstruct_images)
	{
		exp_imagedata.resize(2 * nr_images, mymodel.ori_size, mymodel.ori_size);
	}
	else
	{
		exp_imagedata.resize(nr_images, mymodel.ori_size, mymodel.ori_size);
	}

	for (long int ori_part_id = first_ori_particle_id, my_image_no = 0; ori_part_id <= last_ori_particle_id; ori_part_id++)
	{
		for (long int i = 0; i < mydata.ori_particles[ori_part_id].particles_id.size(); i++)
		{
			long int part_id = mydata.ori_particles[ori_part_id].particles_id[i];
			for (int iseries = 0; iseries < mydata.getNrImagesInSeries(part_id); iseries++, my_image_no++)
			{
				long int img_id = mydata.getImageId(part_id, iseries);

#ifdef DEBUG_CHECKSIZES
				if (img_id >= mydata.MDimg.numberOfObjects())
				{
					std::cerr << "img_id= " << img_id << " mydata.MDimg.numberOfObjects()= " << mydata.MDimg.numberOfObjects() << std::endl;
					REPORT_ERROR("img_id >= mydata.MDimg.numberOfObjects()");
				}
				if (my_image_no >= YSIZE(exp_metadata))
				{
					std::cerr << "my_image_no= " << my_image_no << " YSIZE(exp_metadata)= " << YSIZE(exp_metadata) << std::endl;
					REPORT_ERROR("my_image_no >= YSIZE(exp_metadata)");
				}
				if (my_image_no >= nr_images)
				{
					std::cerr << "my_image_no= " << my_image_no << " nr_images= " << nr_images << std::endl;
					REPORT_ERROR("my_image_no >= nr_images");
				}
#endif
				// First read the image from disc
				FileName fn_img, fn_rec_img;
				mydata.MDimg.getValue(EMDL_IMAGE_NAME, fn_img, img_id);
				Image<DOUBLE> img, rec_img;
				img.read(fn_img);
				if (XSIZE(img()) != XSIZE(exp_imagedata) || YSIZE(img()) != YSIZE(exp_imagedata))
				{
					std::cerr << " fn_img= " << fn_img << " XSIZE(img())= " << XSIZE(img()) << " YSIZE(img())= " << YSIZE(img()) <<" " << XSIZE(exp_imagedata) << " "<< YSIZE(exp_imagedata)<< std::endl;
					REPORT_ERROR("MlOptimiser::getMetaAndImageDataSubset ERROR: incorrect image size");
				}
				if (has_converged && do_use_reconstruct_images)
				{
					mydata.MDimg.getValue(EMDL_IMAGE_RECONSTRUCT_NAME, fn_rec_img, img_id);
					rec_img.read(fn_rec_img);
					if (XSIZE(rec_img()) != XSIZE(exp_imagedata) || YSIZE(rec_img()) != YSIZE(exp_imagedata))
					{
						std::cerr << " fn_rec_img= " << fn_rec_img << " XSIZE(rec_img())= " << XSIZE(rec_img()) << " YSIZE(rec_img())= " << YSIZE(rec_img()) << std::endl;
						REPORT_ERROR("MlOptimiser::getMetaAndImageDataSubset ERROR: incorrect reconstruct_image size");
					}
				}
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(img())
				{
					DIRECT_A3D_ELEM(exp_imagedata, my_image_no, i, j) = DIRECT_A2D_ELEM(img(), i, j);
				}

				if (has_converged && do_use_reconstruct_images)
				{
					FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(rec_img())
					{
						DIRECT_A3D_ELEM(exp_imagedata, nr_images + my_image_no, i, j) = DIRECT_A2D_ELEM(rec_img(), i, j);
					}
				}

				// Now get the metadata
				int iaux;
				mydata.MDimg.getValue(EMDL_ORIENT_ROT,  DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT), img_id);
				mydata.MDimg.getValue(EMDL_ORIENT_TILT, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT), img_id);
				mydata.MDimg.getValue(EMDL_ORIENT_PSI,  DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI), img_id);
				mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF), img_id);
				mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF), img_id);
				mydata.MDimg.getValue(EMDL_PARTICLE_CLASS, iaux, img_id);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CLASS) = (DOUBLE)iaux;
				mydata.MDimg.getValue(EMDL_PARTICLE_DLL,  DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_DLL), img_id);
				mydata.MDimg.getValue(EMDL_PARTICLE_PMAX, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PMAX), img_id);
				mydata.MDimg.getValue(EMDL_PARTICLE_NR_SIGNIFICANT_SAMPLES, iaux, img_id);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NR_SIGN) = (DOUBLE)iaux;
				if (!mydata.MDimg.getValue(EMDL_IMAGE_NORM_CORRECTION, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM), img_id))
				{
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_NORM) = 1.;
				}
				if (do_ctf_correction)
				{
					long int mic_id = mydata.getMicrographId(part_id, iseries);
					DOUBLE kV, DeltafU, DeltafV, azimuthal_angle, Cs, Bfac, Q0;
					if (!mydata.MDimg.getValue(EMDL_CTF_VOLTAGE, kV, img_id))
						if (!mydata.MDmic.getValue(EMDL_CTF_VOLTAGE, kV, mic_id))
						{
							kV = 200;
						}

					if (!mydata.MDimg.getValue(EMDL_CTF_DEFOCUSU, DeltafU, img_id))
						if (!mydata.MDmic.getValue(EMDL_CTF_DEFOCUSU, DeltafU, mic_id))
						{
							DeltafU = 0;
						}

					if (!mydata.MDimg.getValue(EMDL_CTF_DEFOCUSV, DeltafV, img_id))
						if (!mydata.MDmic.getValue(EMDL_CTF_DEFOCUSV, DeltafV, mic_id))
						{
							DeltafV = DeltafU;
						}

					if (!mydata.MDimg.getValue(EMDL_CTF_DEFOCUS_ANGLE, azimuthal_angle, img_id))
						if (!mydata.MDmic.getValue(EMDL_CTF_DEFOCUS_ANGLE, azimuthal_angle, mic_id))
						{
							azimuthal_angle = 0;
						}

					if (!mydata.MDimg.getValue(EMDL_CTF_CS, Cs, img_id))
						if (!mydata.MDmic.getValue(EMDL_CTF_CS, Cs, mic_id))
						{
							Cs = 0;
						}

					if (!mydata.MDimg.getValue(EMDL_CTF_BFACTOR, Bfac, img_id))
						if (!mydata.MDmic.getValue(EMDL_CTF_BFACTOR, Bfac, mic_id))
						{
							Bfac = 0;
						}

					if (!mydata.MDimg.getValue(EMDL_CTF_Q0, Q0, img_id))
						if (!mydata.MDmic.getValue(EMDL_CTF_Q0, Q0, mic_id))
						{
							Q0 = 0;
						}

					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_VOLTAGE) = kV;
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_DEFOCUS_U) = DeltafU;
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_DEFOCUS_V) = DeltafV;
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_DEFOCUS_ANGLE) = azimuthal_angle;
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_CS) = Cs;
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_BFAC) = Bfac;
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_CTF_Q0) = Q0;

				}

				// beamtilt
				DOUBLE beamtilt_x = 0., beamtilt_y = 0.;
				if (mydata.MDimg.containsLabel(EMDL_IMAGE_BEAMTILT_X))
				{
					mydata.MDimg.getValue(EMDL_IMAGE_BEAMTILT_X, beamtilt_x, img_id);
				}
				if (mydata.MDimg.containsLabel(EMDL_IMAGE_BEAMTILT_Y))
				{
					mydata.MDimg.getValue(EMDL_IMAGE_BEAMTILT_Y, beamtilt_y, img_id);
				}
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_BEAMTILT_X) = beamtilt_x;
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_BEAMTILT_Y) = beamtilt_y;

				// If the priors are NOT set, then set their values to 999.
				if (!mydata.MDimg.getValue(EMDL_ORIENT_ROT_PRIOR,  DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT_PRIOR), img_id))
				{
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT_PRIOR) = 999.;
				}
				if (!mydata.MDimg.getValue(EMDL_ORIENT_TILT_PRIOR, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT_PRIOR), img_id))
				{
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT_PRIOR) = 999.;
				}
				if (!mydata.MDimg.getValue(EMDL_ORIENT_PSI_PRIOR,  DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI_PRIOR), img_id))
				{
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI_PRIOR) = 999.;
				}
				if (!mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_X_PRIOR, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF_PRIOR), img_id))
				{
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF_PRIOR) = 999.;
				}
				if (!mydata.MDimg.getValue(EMDL_ORIENT_ORIGIN_Y_PRIOR, DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF_PRIOR), img_id))
				{
					DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF_PRIOR) = 999.;
				}

				// Pass the transformation matrix (even if it is the Identity matrix...
				Matrix2D<DOUBLE> R_mic;
				R_mic = mydata.getMicrographTransformationMatrix(part_id, iseries);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_0) = R_mic(0, 0);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_1) = R_mic(0, 1);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_0_2) = R_mic(0, 2);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_0) = R_mic(1, 0);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_1) = R_mic(1, 1);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_1_2) = R_mic(1, 2);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_0) = R_mic(2, 0);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_1) = R_mic(2, 1);
				DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_MAT_2_2) = R_mic(2, 2);

			}
		}
	}

}

void MlOptimiser::do_nothing()
{
	exp_ipart_ThreadTaskDistributor->reset();
	global_ThreadManager->run(globalGetFourierTransformsAndCtfs);
}
void MlOptimiser:: subYellowMK_gpu( DOUBLE *image_red_D)
{
	
		Matrix2D<DOUBLE> A3D, Ainv;
		MultidimArray<DOUBLE>  dummy;
	
		DOUBLE rot, tilt,  psi ,xoff, yoff, zoff;
		DOUBLE *A3D_D, *A3D_H;
		DOUBLE angpix, maxres;
		angpix = mymodel.pixel_size;
		maxres = -1;
		DOUBLE r_max;
		CUFFT_COMPLEX *Fref_all_yellow_D, *Fref_all_shift_D, *Fref_all_yellow_H;
		DOUBLE *Yellow_imgs_D, *project_imgs_D;
		A3D_H = (DOUBLE*) malloc(sizeof(DOUBLE)* exp_nr_images* 9);
		cudaMalloc((void**)&A3D_D, sizeof(DOUBLE)* exp_nr_images* 9);
		
		if (maxres < 0.)
				r_max = mymodel.Iref_yellow[0].xdim;
		else
				r_max = CEIL(mymodel.Iref_yellow[0].xdim* angpix /maxres);
				
		cudaError cu_error = cudaGetLastError();
		int  f2d_x, f2d_y, data_x, data_y, data_z, data_starty, data_startz;
		f2d_x = mymodel.Iref_yellow[0].xdim/2+1;
		f2d_y = mymodel.Iref_yellow[0].ydim;
		int image_size;
		image_size = f2d_x * f2d_y;
		data_x = XSIZE(mymodel.PPref_yellow[0].data);
		data_y = YSIZE(mymodel.PPref_yellow[0].data);
		data_z = ZSIZE(mymodel.PPref_yellow[0].data);
	
		data_starty = STARTINGY(mymodel.PPref_yellow[0].data);
		data_startz = STARTINGZ(mymodel.PPref_yellow[0].data);
		//std::cout << data_x << "  " << data_starty << std::endl;
 		//The only one is not necessory in GPU implementation
				
			MetaDataTable DFo, MDang;
			//Matrix2D<DOUBLE> A3D;
			FileName fn_expimg;
	
			MultidimArray<Complex > F3D, F2D, Fexpimg;
			MultidimArray<DOUBLE> Fctf;
			Image<DOUBLE> vol, img, expimg;
			FourierTransformer transformer, transformer_expimg;
	
			img().resize(YSIZE(mymodel.yellow_mask()), XSIZE(mymodel.yellow_mask()));
    		//transformer.setReal(img());
    		//transformer.getFourierAlias(F2D);
			  
		  	// Set up the projector
		  	//Projector projector=mymodel.PPref_yellow[0];
			//compare_CPU_GPU(projector.data.data, project_data_D + 2*(mymodel.PPref[0]).data.zyxdim*mymodel.nr_classes, (mymodel.PPref_yellow[0]).data.zyxdim, "yellow data", true); 	
		
		  	//projector.computeFourierTransformMap(vol(), dummy, 2* r_max);
			 //==================================================================			 
			rot = tilt = psi = xoff = yoff = zoff = 0.;
	
			//DO not support add noise
			
			 DOUBLE *shift_H;
			 DOUBLE *Fctf_H;
			 shift_H  = (DOUBLE *) malloc(exp_nr_images*2*sizeof(DOUBLE));
			 bool *shift_flag_H, *shift_flag_D;
			 shift_flag_H = (bool*) malloc(sizeof(bool) * exp_nr_images);
			 cudaMalloc((void **)&shift_flag_D, sizeof(bool) * exp_nr_images);
			 Fref_all_yellow_H = (CUFFT_COMPLEX*)malloc(exp_nr_images * image_size * sizeof(CUFFT_COMPLEX ));
			 for(int my_image_no =0; my_image_no< exp_nr_images; my_image_no++)
			 {
				rot = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ROT); 
				tilt = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_TILT); 
				psi = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_PSI); 
				
				xoff = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_XOFF); 
				yoff = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_YOFF); 
				zoff = DIRECT_A2D_ELEM(exp_metadata, my_image_no, METADATA_ZOFF); 
	
				Euler_rotation3DMatrix(rot, tilt, psi, A3D);
				
				for(int i=0; i < 3; i++){
					for(int j=0; j < 3; j++)
					{
						A3D_H[my_image_no*9+i*3+j]=A3D.mdata[i*4+j];
					}
				}
				//F2D.initZeros();
            			//projector.get2DFourierTransform(F2D, A3D, IS_NOT_INV);
				//memcpy(Fref_all_yellow_H+image_size*my_image_no, F2D.data, image_size*sizeof(CUFFT_COMPLEX ));
				
				if (ABS(xoff) > 0.001 || ABS(yoff) > 0.001)
				{
					shift_H[my_image_no*2] = -xoff;
					shift_H[my_image_no*2+1] = -yoff;
					shift_flag_H[my_image_no] = true;
				}
				else
				{
					shift_flag_H[my_image_no] = false;
				}
				
			 }
			 cudaMemcpy(A3D_D, A3D_H, exp_nr_images*9*sizeof(DOUBLE), cudaMemcpyHostToDevice);
			 cudaMemcpy(shift_flag_D, shift_flag_H, exp_nr_images*sizeof(bool), cudaMemcpyHostToDevice);
	
			 cudaMalloc((void**)&Fref_all_yellow_D, exp_nr_images * image_size * sizeof(CUFFT_COMPLEX ));
			 cudaMemset(Fref_all_yellow_D, 0., exp_nr_images * image_size * sizeof(CUFFT_COMPLEX ));
			 
			 cudaMalloc((void **)&Fref_all_shift_D, exp_nr_images* image_size * sizeof(CUFFT_COMPLEX ));
			 cudaDeviceSynchronize();
			 cu_error = cudaGetLastError();
			 if (cu_error != cudaSuccess)
			 {
				std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
				REPORT_ERROR("cudaKernel get2DFourierTransform_gpu subMK ");
			 } 
			 //std::cout<< "f2Dx : " << f2d_x<<"  mymodel.ori_size : " << mymodel.ori_size<<  " mymodel.PPref[0].data " << mymodel.PPref[0].data.zyxdim<< std::endl;
			 //std::cout <<"data_x " << data_x << "  data_y " << data_y << " data_z " << data_z << std::endl;
			 CUFFT_COMPLEX *yellow_project_D = project_data_D + 2*(mymodel.PPref[0]).data.zyxdim*mymodel.nr_classes;
			 get2DFourierTransform_gpu(Fref_all_yellow_D,
											  A3D_D,
											  yellow_project_D,
											  IS_NOT_INV,
											  mymodel.PPref_yellow[0].padding_factor,
											  mymodel.PPref_yellow[0].r_max,
											  mymodel.PPref_yellow[0].r_min_nn,
											  f2d_x, 
											  f2d_y,
											  data_x,
											  data_y,
											  data_z,
											  data_starty,
											  data_startz,
											  exp_nr_images,
											  mymodel.ref_dim);
			 
			 
		cudaDeviceSynchronize();
		
		 Fctf_H= (DOUBLE *)malloc( exp_nr_images * image_size * sizeof(DOUBLE));
		
		cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel get2DFourierTransform_gpu subMK ");
		}
			//cudaMemcpy(Fref_all_yellow_D, Fref_all_yellow_H, (exp_nr_images)*image_size*sizeof(CUFFT_COMPLEX), cudaMemcpyHostToDevice);
			shiftImageInFourierTransform_gpu(Fref_all_yellow_D,
											 Fref_all_shift_D,
											 (DOUBLE) mymodel.ori_size, shift_H,
											 shift_flag_D,
											 exp_nr_images, 
											 f2d_x, f2d_y, 1);
			
			cudaDeviceSynchronize();
		//compare_CPU_GPU(F2D.data,Fref_all_shift_D+(exp_nr_images-1)*image_size, image_size ,"Fref_all_shift_D Fourier", true);
		free(Fref_all_yellow_H);
		
		cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel shiftImageInFourierTransform_gpu subMK ");
		}
		DOUBLE* ctf_related_parameters_H;
		int nr_parameters = 9;
		ctf_related_parameters_H = (DOUBLE*) malloc(nr_parameters * exp_nr_images * sizeof(DOUBLE));
		for (int im = 0; im < exp_nr_images; im++)
		{
		
			DOUBLE DeltafU = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_DEFOCUS_U);
			DOUBLE DeltafV = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_DEFOCUS_V);
			DOUBLE azimuthal_angle	= DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_DEFOCUS_ANGLE);
			DOUBLE kV	= DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_VOLTAGE);
			DOUBLE Cs	= DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_CS);
			DOUBLE Q0	= DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_Q0);
			DOUBLE Bfac = DIRECT_A2D_ELEM(exp_metadata, im, METADATA_CTF_BFAC);
			DOUBLE scale = 1.0;//DIRECT_A2D_ELEM(exp_metadata, im, EMDL_CTF_SCALEFACTOR1);//EMDL_CTF_SCALEFACTOR1.0;
			DOUBLE local_Cs = Cs * 1e7;
			DOUBLE local_kV = kV * 1e3;
			DOUBLE rad_azimuth = DEG2RAD(azimuthal_angle);
	
			DOUBLE defocus_average	 = -(DeltafU + DeltafV) * 0.5;
			DOUBLE defocus_deviation = -(DeltafU - DeltafV) * 0.5;
	
			DOUBLE lambda = 12.2643247 / sqrt(local_kV * (1. + local_kV * 0.978466e-6));
	
			DOUBLE K1 = PI / 2 * 2 * lambda;
			DOUBLE K2 = PI / 2 * local_Cs * lambda * lambda * lambda;
			DOUBLE K3 = sqrt(1 - Q0 * Q0);
			DOUBLE K4 = -Bfac/4.;
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
		DOUBLE *Fctf_D;
		cudaMalloc((void**)&Fctf_D, exp_nr_images * image_size * sizeof(DOUBLE));
		getFftwImage_gpu(Fctf_D, ctf_related_parameters_H,
						 mymodel.ori_size, mymodel.ori_size, angpix, //mymodel.pixel_size,
						 ctf_phase_flipped, false, false, true,
						 exp_nr_images,
						 f2d_x,
						 f2d_y,
						 do_ctf_correction);
		
		cudaDeviceSynchronize();
		//compare_CPU_GPU(Fctf.data,Fctf_D+(exp_nr_images-1)*image_size, image_size ,"Fctf", true);
		//cudaMemcpy(Fctf_D, Fctf_H,exp_nr_images * image_size * sizeof(DOUBLE), cudaMemcpyHostToDevice );
		cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel getFftwImage_gpu subMK ");
		}
		free(ctf_related_parameters_H);
		free(Fctf_H);
		
		apply_CTF_gpu(Fref_all_shift_D, Fctf_D, exp_nr_images, image_size);
		cudaDeviceSynchronize();
		
		//compare_CPU_GPU(F2D.data,Fref_all_shift_D+(exp_nr_images-1)*image_size, image_size ,"Fref_all_shift_D Fourier", true);
		//compare_CPU_GPU(F2D.data,Fref_all_shift_D+(exp_nr_images-1)*image_size, image_size ,"Fref_all_shift_D", true);
		cu_error = cudaGetLastError();
		 if (cu_error != cudaSuccess)
		{
			 std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel shiftImageInFourierTransform_gpu subMK ");
		}  
		cudaMalloc((void**)&project_imgs_D, exp_nr_images* sizeof(DOUBLE) * mymodel.Iref_yellow[0].xdim*mymodel.Iref_yellow[0].ydim);
		cudaMalloc((void**)&Yellow_imgs_D, exp_nr_images* sizeof(DOUBLE) * mymodel.Iref_yellow[0].xdim*mymodel.Iref_yellow[0].ydim);
		//cudaMemset(Yellow_imgs_D, 0., exp_nr_images* sizeof(DOUBLE) * mymodel.Iref_yellow[0].xdim*mymodel.Iref_yellow[0].ydim);
		int ndim = 2;
		int *N;
		N = (int*)malloc(sizeof(int) * ndim);
		N[0] = mymodel.Iref_yellow[0].ydim;
		N[1] = mymodel.Iref_yellow[0].xdim;
		
		cufftHandle fPlanBackward_gpu;
#ifdef FLOAT_PRECISION
		cufftResult fftplan1 = cufftPlanMany(&fPlanBackward_gpu ,	ndim, N,
										NULL, 1, 0,
										NULL, 1, 0,
										CUFFT_C2R, exp_nr_images);
		cufftExecC2R(fPlanBackward_gpu,  Fref_all_shift_D, Yellow_imgs_D);	
#else
		cufftResult fftplan1 = cufftPlanMany(&fPlanBackward_gpu, ndim, N,
											 NULL, 1, 0,
											 NULL, 1, 0,
											 CUFFT_Z2D, exp_nr_images);
		cufftExecZ2D(fPlanBackward_gpu,  Fref_all_shift_D, Yellow_imgs_D);
#endif
		cufftDestroy(fPlanBackward_gpu);
		cudaDeviceSynchronize();
		cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel centerFFT_gpu subMK ");
		}
		
		//centerFFT_gpu(Yellow_imgs_D , project_imgs_D, exp_nr_images, ndim, mymodel.Iref_yellow[0].xdim,  mymodel.Iref_yellow[0].ydim, 1, false);
		centerFFT_2_gpu(Yellow_imgs_D , project_imgs_D, exp_nr_images, ndim, mymodel.Iref_yellow[0].xdim,	mymodel.Iref_yellow[0].ydim, 1, false);
		cudaDeviceSynchronize();
		cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel centerFFT_gpu subMK ");
		}
		/*DOUBLE *image_H;
		image_H = (DOUBLE*) malloc( mymodel.Iref_yellow[0].xdim*mymodel.Iref_yellow[0].ydim*sizeof(DOUBLE));
		cudaMemcpy(image_H, project_imgs_D,  mymodel.Iref_yellow[0].xdim*mymodel.Iref_yellow[0].ydim*sizeof(DOUBLE), cudaMemcpyDeviceToHost);

		for(int i =0; i < mymodel.Iref_yellow[0].xdim*mymodel.Iref_yellow[0].ydim; i++)
		{
			if(image_H[i]!=0)
			{
				std::cout<< "none zero value "<< image_H[i]<< "  " << i << std::endl;
				
				REPORT_ERROR("none zero value");
			}
		}*/
		//cudaMemset(project_imgs_D, 0., mymodel.Iref_yellow[0].xdim*mymodel.Iref_yellow[0].ydim*sizeof(DOUBLE)*exp_nr_images);
		sub_Yellow_project_gpu(image_red_D, project_imgs_D, exp_nr_images,	mymodel.Iref_yellow[0].xdim,  mymodel.Iref_yellow[0].ydim);
		
		cudaDeviceSynchronize();
		//compare_CPU_GPU(image_H,image_red_D, mymodel.Iref_yellow[0].xdim*mymodel.Iref_yellow[0].ydim ,"image_H ", true);

		//free(image_H);
		//cudaDeviceSynchronize();
		cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel sub_Yellow_project_gpu subMK ");
		}
		//sub_extract = false;
		cudaDeviceSynchronize();
		cudaFree(project_imgs_D);
		cudaFree(Yellow_imgs_D);
		cudaFree(Fctf_D);
		cudaFree(shift_flag_D);
		cudaFree(Fref_all_yellow_D);
		cudaFree(A3D_D);
		cudaFree(Fref_all_shift_D);
		
		free(N);
		free(shift_H);
		free(shift_flag_H);
		free(A3D_H);
		
		cu_error = cudaGetLastError();
		if (cu_error != cudaSuccess)
		{
			std::cout << "cudaMemcpy Error :" << __LINE__ <<" error : " << cudaGetErrorString(cu_error) << std::endl;
			REPORT_ERROR("cudaKernel free subMK ");
		}
	
	}

