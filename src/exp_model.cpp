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
#include "src/exp_model.h"

void ExpOriginalParticle::addParticle(long int _particle_id, int _random_subset, int _order)
{
	// Keep random_subsets equal in each original particle
	if (random_subset != _random_subset)
		REPORT_ERROR("ExpOriginalParticle:addParticle: incompatible random subsets between particle and its original particle");
	particles_id.push_back(_particle_id);
	particles_order.push_back(_order);
}

	long int Experiment::numberOfImages(int random_subset)
{
	long int result = 0;
	for (long int i = 0; i < particles.size(); i++)
		if (random_subset == 0 || particles[i].random_subset == random_subset)
			result += particles[i].images.size();

	return result;
}

long int Experiment::numberOfParticles(int random_subset)
{
	if (random_subset == 0)
		return particles.size();
	else
	{
		long int result = 0;
		for (long int i = 0; i < ori_particles.size(); i++)
		{
			if (ori_particles[i].random_subset == random_subset)
			{
				result += ori_particles[i].particles_id.size();
			}
		}
		return result;
	}
}

long int Experiment::numberOfOriginalParticles(int random_subset)
{
	if (random_subset == 0)
		return ori_particles.size();
	else if (random_subset == 1)
		return nr_ori_particles_subset1;
	else if (random_subset == 2)
		return nr_ori_particles_subset2;
	else
		REPORT_ERROR("ERROR: Experiment::numberOfOriginalParticles invalid random_subset: " + integerToString(random_subset));
}


long int Experiment::numberOfMicrographs()
{
	return micrographs.size();
}

long int Experiment::numberOfGroups()
{
	return groups.size();
}

int Experiment::getNrImagesInSeries(long int part_id)
{
//#define DEBUG_CHECKSIZES
#ifdef DEBUG_CHECKSIZES
	if (part_id >= particles.size())
	{
		std::cerr<< "part_id= "<<part_id<<" particles.size()= "<< particles.size() <<std::endl;
		REPORT_ERROR("part_id >= particles.size()");
	}
#endif
	return (particles[part_id].images).size();
}

long int Experiment::getMicrographId(long int part_id, int inseries_no)
{
#ifdef DEBUG_CHECKSIZES
	if (part_id >= particles.size())
	{
		std::cerr<< "part_id= "<<part_id<<" particles.size()= "<< particles.size() <<std::endl;
		REPORT_ERROR("part_id >= particles.size()");
	}
	if (inseries_no >= particles[part_id].images.size())
	{
		std::cerr<< "inseries_no= "<<inseries_no<<" particles[part_id].images.size()= "<< particles[part_id].images.size() <<std::endl;
		REPORT_ERROR("inseries_no >= particles[part_id].images.size()]");
	}
#endif
	return (particles[part_id].images[inseries_no]).micrograph_id;
}

long int Experiment::getGroupId(long int part_id, int inseries_no)
{
#ifdef DEBUG_CHECKSIZES
	if (part_id >= particles.size())
	{
		std::cerr<< "part_id= "<<part_id<<" particles.size()= "<< particles.size() <<std::endl;
		REPORT_ERROR("part_id >= particles.size()");
	}
	if (inseries_no >= particles[part_id].images.size())
	{
		std::cerr<< "inseries_no= "<<inseries_no<<" particles[part_id].images.size()= "<< particles[part_id].images.size() <<std::endl;
		REPORT_ERROR("inseries_no >= particles[part_id].images.size()]");
	}
#endif
	return (particles[part_id].images[inseries_no]).group_id;
}

int Experiment::getRandomSubset(long int part_id)
{
	return particles[part_id].random_subset;
}

long int Experiment::getImageId(long int part_id, int inseries_no)
{
	return (particles[part_id].images[inseries_no]).id;
}

MetaDataTable Experiment::getMetaDataImage(long int part_id, int inseries_no)
{
	MetaDataTable result;
	long int img_id = getImageId(part_id, inseries_no);
	result.addObject(MDimg.getObject(img_id));
	return result;
}

MetaDataTable Experiment::getMetaDataMicrograph(long int part_id, int inseries_no)
{
	MetaDataTable result;
	long int mic_id = getMicrographId(part_id, inseries_no);
	result.addObject(MDmic.getObject(mic_id));
	return result;
}

Matrix2D<double> Experiment::getMicrographTransformationMatrix(long int micrograph_id)
{

	Matrix2D<double> R(3,3);

	if (MDmic.containsLabel(EMDL_MATRIX_1_1))
	{

#ifdef DEBUG_CHECKSIZES
	if (micrograph_id >= MDmic.numberOfObjects())
	{
		std::cerr<< "micrograph_id= "<<micrograph_id<<" MDmic.lastObject()= "<< MDmic.lastObject() <<std::endl;
		REPORT_ERROR("micrograph_id > MDmic.lastObject()");
	}
#endif
		//TODO: default values for getValue
		MDmic.getValue(EMDL_MATRIX_1_1, R(0,0), micrograph_id);
		MDmic.getValue(EMDL_MATRIX_1_2, R(0,1), micrograph_id);
		MDmic.getValue(EMDL_MATRIX_1_3, R(0,2), micrograph_id);
		MDmic.getValue(EMDL_MATRIX_2_1, R(1,0), micrograph_id);
		MDmic.getValue(EMDL_MATRIX_2_2, R(1,1), micrograph_id);
		MDmic.getValue(EMDL_MATRIX_2_3, R(1,2), micrograph_id);
		MDmic.getValue(EMDL_MATRIX_3_1, R(2,0), micrograph_id);
		MDmic.getValue(EMDL_MATRIX_3_2, R(2,1), micrograph_id);
		MDmic.getValue(EMDL_MATRIX_3_3, R(2,2), micrograph_id);
	}
	else if (MDmic.containsLabel(EMDL_MICROGRAPH_TILT_ANGLE))
	{
#ifdef DEBUG_CHECKSIZES
	if (micrograph_id >= MDmic.numberOfObjects())
	{
		std::cerr<< "micrograph_id= "<<micrograph_id<<" MDmic.lastObject()= "<< MDmic.lastObject() <<std::endl;
		REPORT_ERROR("micrograph_id > MDmic.lastObject()");
	}
#endif

	double tiltdir, tiltangle, outofplane;
		MDmic.getValue(EMDL_MICROGRAPH_TILT_ANGLE, tiltangle, micrograph_id);
		// By default tiltdir = 0
		if (!MDmic.getValue(EMDL_MICROGRAPH_TILT_AXIS_DIRECTION, tiltdir, micrograph_id))
			tiltdir = 0.;
		// By default in-plane tilt axis....
		if (!MDmic.getValue(EMDL_MICROGRAPH_TILT_AXIS_OUTOFPLANE, outofplane, micrograph_id))
			outofplane= 90.;

		// Transformation matrix
		// Get the direction of the tilt axis as a Matrix1D
		Matrix1D<double> dir_axis(3);
		Euler_angles2direction(tiltdir, outofplane, dir_axis);
		// Calculate the corresponding 3D rotation matrix
		rotation3DMatrix(tiltangle, dir_axis, R, false);
		// Somehow have to take the inverse of that...
		R = R.inv();
	}
	else
	{
		R.initIdentity();
	}

	return R;
}

Matrix2D<double> Experiment::getMicrographTransformationMatrix(long int part_id, int inseries_no)
{
	long int mic_id = getMicrographId(part_id, inseries_no);
	return getMicrographTransformationMatrix(mic_id);
}


long int Experiment::addImage(long int group_id, long int micrograph_id, long int particle_id)
{

	if (group_id >= groups.size())
		REPORT_ERROR("Experiment::addImage: group_id out of range");

	if (micrograph_id >= micrographs.size())
		REPORT_ERROR("Experiment::addImage: micrograph_id out of range");

	if (particle_id >= particles.size())
	{
		std::cerr << " particle_id= " << particle_id << " particles.size()= " << particles.size() << std::endl;
		REPORT_ERROR("Experiment::addImage: particle_id out of range");
	}

	ExpImage image;
	image.group_id = group_id;
	image.micrograph_id = micrograph_id;
	image.particle_id = particle_id;
	// Add new entry in the MDimg MetaDataTable of this Experiment, and get image.id
	image.id = MDimg.addObject();

	// add this ExpImage to the ExpParticle and the ExpMicrograph
	(micrographs[micrograph_id].images).push_back(image);
	(particles[particle_id].images).push_back(image);

	return image.id;

}

long int Experiment::addParticle(std::string part_name, int random_subset)
{

	ExpParticle particle;
	particle.id = particles.size();
	particle.random_subset = random_subset;
	particle.name = part_name;
	// Push back this particle in the particles vector
	particles.push_back(particle);

	// Return the id in the particles vector
	return particle.id;

}

long int Experiment::addOriginalParticle(std::string part_name, int _random_subset)
{

	ExpOriginalParticle ori_particle;
	ori_particle.random_subset = _random_subset;
	ori_particle.name = part_name;
	long int id = ori_particles.size();
	ori_particles.push_back(ori_particle);

	// Return the id in the ori_particles vector
	return id;

}

long int Experiment::addGroup(std::string group_name)
{
	// Add new entry in the MDmic MetaDataTable of this Experiment
	ExpGroup group;
	group.id = groups.size(); // start counting groups at 0!
	group.name = group_name;

	// Push back this micrograph
	groups.push_back(group);

	// Return the id in the micrographs vector
	return group.id;

}

long int Experiment::addAverageMicrograph(std::string avg_mic_name)
{
	// Add new entry in the MDmic MetaDataTable of this Experiment
	AverageMicrograph micrograph;
	micrograph.id = average_micrographs.size();
	micrograph.name = avg_mic_name;

	// Push back this micrograph
	average_micrographs.push_back(micrograph);

	// Return the id in the micrographs vector
	return micrograph.id;

}

long int Experiment::addMicrograph(std::string mic_name)
{
	// Add new entry in the MDmic MetaDataTable of this Experiment
	ExpMicrograph micrograph;
	micrograph.id = MDmic.addObject();
	micrograph.name = mic_name;

	// Push back this micrograph
	micrographs.push_back(micrograph);

	// Return the id in the micrographs vector
	return micrograph.id;

}

void Experiment::divideOriginalParticlesInRandomHalves(int seed)
{

	// Only do this if the random_subset of all original_particles is zero
	bool all_are_zero = true;
	bool some_are_zero = false;
	nr_ori_particles_subset1 = 0;
	nr_ori_particles_subset2 = 0;
	for (long int i = 0; i < ori_particles.size(); i++)
	{
		int random_subset = ori_particles[i].random_subset;
		if (random_subset != 0)
		{
			all_are_zero = false;
			// Keep track of how many particles there are in each subset
			if (random_subset == 1)
				nr_ori_particles_subset1++;
			else if (random_subset == 2)
				nr_ori_particles_subset2++;
			else
				REPORT_ERROR("ERROR Experiment::divideParticlesInRandomHalves: invalid number for random subset (i.e. not 1 or 2): " + integerToString(random_subset));
		}
		else
			some_are_zero = true;

		if (!all_are_zero && some_are_zero)
			REPORT_ERROR("ERROR Experiment::divideParticlesInRandomHalves: some random subset values are zero and others are not. They should all be zero, or all bigger than zero!");
	}

	//std::cerr << " all_are_zero= " << all_are_zero << " some_are_zero= " << some_are_zero << std::endl;

	if (all_are_zero)
	{
		// Only randomise them if the random_subset values were not read in from the STAR file
		srand(seed);
		for (long int i = 0; i < ori_particles.size(); i++)
		{
			int random_subset = rand() % 2 + 1;
			ori_particles[i].random_subset = random_subset; // randomly 1 or 2
			if (random_subset == 1)
				nr_ori_particles_subset1++;
			else if (random_subset == 2)
				nr_ori_particles_subset2++;
			else
				REPORT_ERROR("ERROR Experiment::divideParticlesInRandomHalves: invalid number for random subset (i.e. not 1 or 2): " + integerToString(random_subset));

			// Loop over all particles in each ori_particle and set their random_subset
			// Also set the EMDL_PARTICLE_RANDOM_SUBSET in the MDimg of all images for this particle
			for (long int j = 0; j < ori_particles[i].particles_id.size(); j++)
			{
				long int part_id = (ori_particles[i]).particles_id[j];
				{
					particles[part_id].random_subset = random_subset;
					for (long int n = 0; n < (particles[part_id]).images.size(); n++)
					{
						long int img_id = ((particles[part_id]).images[n]).id;
						MDimg.setValue(EMDL_PARTICLE_RANDOM_SUBSET, random_subset, img_id);
					}
				}
			}
		}
	}
}

void Experiment::randomiseOriginalParticlesOrder(int seed, bool do_split_random_halves)
{
	//This static flag is for only randomize once
	static bool randomised = false;
	if (!randomised)
	{

		srand(seed);
		std::vector<ExpOriginalParticle> new_ori_particles;

		if (do_split_random_halves)
		{
			std::vector<long int> ori_particle_list1, ori_particle_list2;
			ori_particle_list1.clear();
			ori_particle_list2.clear();
			// Fill the two particle lists
			for (long int i = 0; i < ori_particles.size(); i++)
			{
				int random_subset = ori_particles[i].random_subset;
				if (random_subset == 1)
					ori_particle_list1.push_back(i);
				else if (random_subset == 2)
					ori_particle_list2.push_back(i);
				else
					REPORT_ERROR("ERROR Experiment::randomiseParticlesOrder: invalid number for random subset (i.e. not 1 or 2): " + integerToString(random_subset));
			}

			// Just a silly check for the sizes of the ori_particle_lists (to be sure)
			if (ori_particle_list1.size() != nr_ori_particles_subset1)
				REPORT_ERROR("ERROR Experiment::randomiseParticlesOrder: invalid ori_particle_list1 size:" + integerToString(ori_particle_list1.size()) + " != " + integerToString(nr_ori_particles_subset1));
			if (ori_particle_list2.size() != nr_ori_particles_subset2)
				REPORT_ERROR("ERROR Experiment::randomiseParticlesOrder: invalid ori_particle_list2 size:" + integerToString(ori_particle_list2.size()) + " != " + integerToString(nr_ori_particles_subset2));

			// Randomise the two particle lists
			std::random_shuffle(ori_particle_list1.begin(), ori_particle_list1.end());
			std::random_shuffle(ori_particle_list2.begin(), ori_particle_list2.end());

			// First fill new_ori_particles with the first subset, then with the second
			for (long int i = 0; i < ori_particle_list1.size(); i++)
				new_ori_particles.push_back(ori_particles[ori_particle_list1[i]]);
			for (long int i = 0; i < ori_particle_list2.size(); i++)
				new_ori_particles.push_back(ori_particles[ori_particle_list2[i]]);

		}
		else
		{

			// First fill in order
			std::vector<long int> ori_particle_list;
			ori_particle_list.resize(ori_particles.size());
			for (long int i = 0; i < ori_particle_list.size(); i++)
				ori_particle_list[i] = i;

			// Randomise
			std::random_shuffle(ori_particle_list.begin(), ori_particle_list.end());

			// Refill new_ori_particles
			for (long int i = 0; i < ori_particle_list.size(); i++)
				new_ori_particles.push_back(ori_particles[ori_particle_list[i]]);
		}

		ori_particles=new_ori_particles;
		randomised = true;

	}
}

int Experiment::maxNumberOfImagesPerOriginalParticle(long int first_ori_particle_id, long int last_ori_particle_id)
{

	// By default search all particles
	if (first_ori_particle_id < 0)
		first_ori_particle_id = 0;
	if (last_ori_particle_id < 0)
		last_ori_particle_id = ori_particles.size() - 1;

	int result = 0;
	for (long int i = first_ori_particle_id; i <= last_ori_particle_id; i++)
	{
		// Loop over all particles in this ori_particle
		for (int j = 0; j < ori_particles[i].particles_id.size(); j++)
		{
			long int part_id = ori_particles[i].particles_id[j];
			int val = getNrImagesInSeries(part_id);
			if (val > result)
				result = val;

		}
	}
	return result;

}
void Experiment::expandToMovieFrames(FileName fn_data_movie)
{

	MetaDataTable MDmovie;
	MDmovie.read(fn_data_movie);
	if (!MDmovie.containsLabel(EMDL_MICROGRAPH_NAME) || !MDmovie.containsLabel(EMDL_PARTICLE_NAME))
		REPORT_ERROR("Experiment::expandToMovieFrames Error: movie metadata file does not contain rlnMicrographName as well as rlnParticleName");

	// Re-build new Experiment Exp_movie from scratch
	Experiment Exp_movie;

	// Make a temporary vector of all image names in the current Experiment to gain speed
	std::vector<FileName> fn_curr_imgs, fn_curr_groups;
	std::vector<int> count_frames;
	std::vector<long int> pointer_current_idx;
	FileName fn_curr_img;
	FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDimg)
	{
		MDimg.getValue(EMDL_IMAGE_NAME, fn_curr_img);
		long int group_id;
		MDimg.getValue(EMDL_MLMODEL_GROUP_NO, group_id);
		fn_curr_imgs.push_back(fn_curr_img);
		fn_curr_groups.push_back(groups[group_id-1].name);
		count_frames.push_back(0);
	}

	FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDmovie)
	{
		long int group_id, mic_id, part_id, image_id;
		int my_random_subset, my_class;
		double rot, tilt, psi, xoff, yoff;
		FileName fn_movie_part, fn_curr_img, group_name;
		MDmovie.getValue(EMDL_PARTICLE_NAME, fn_movie_part);

		bool have_found = false;
		for (long int idx = 0; idx < fn_curr_imgs.size(); idx++)
		{
			// Found a match
			if (fn_curr_imgs[idx] == fn_movie_part)
			{
				// Now get the angles from the current Experiment
				MDimg.getValue(EMDL_ORIENT_ROT, rot, idx);
				MDimg.getValue(EMDL_ORIENT_TILT, tilt, idx);
				MDimg.getValue(EMDL_ORIENT_PSI, psi, idx);
				MDimg.getValue(EMDL_ORIENT_ORIGIN_X, xoff, idx);
				MDimg.getValue(EMDL_ORIENT_ORIGIN_Y, yoff, idx);
				MDimg.getValue(EMDL_PARTICLE_CLASS, my_class, idx);
				// Also get the random subset (if present)
				if (!MDimg.getValue(EMDL_PARTICLE_RANDOM_SUBSET, my_random_subset, idx))
					my_random_subset = 0;

				// count how many frames are measured for each particle
				count_frames[idx]++;
				// Also keep track to which particle each image in MDmovie belongs
				pointer_current_idx.push_back(idx);
				group_name = fn_curr_groups[idx];
				have_found = true;
				break;
			}

		}

		// Only include particles that were already in the current Experiment
		if (have_found)
		{
			// Add new micrographs or get mic_id for existing micrograph
			FileName mic_name;
			MDmovie.getValue(EMDL_MICROGRAPH_NAME, mic_name);

			// If this micrograph did not exist in the Exp_movie yet, add it to the Exp_movie experiment
			mic_id = -1;
			for (long int i = 0; i < Exp_movie.micrographs.size(); i++)
			{
				if (Exp_movie.micrographs[i].name == mic_name)
				{
					mic_id = Exp_movie.micrographs[i].id;
					break;
				}
			}
			if (mic_id < 0)
				mic_id = Exp_movie.addMicrograph(mic_name);


			// Add frameno@ to existing group names, so that separate weighting may be applied to different dose images
			// NO THIS HAS NO SENSE IF WE'RE ONLY DOING ONE ITERATION ANYWAY!!! THEN IT'S JUST A WASTE OF MEMORY....
			//std::string dum;
			//long int frameno;
			//mic_name.decompose(frameno, dum);
			//group_name.compose(frameno, group_name, 4);

			// If this group did not exist yet, add it to the experiment
			group_id = -1;
			for (long int i = 0; i < Exp_movie.groups.size(); i++)
			{
				if (Exp_movie.groups[i].name == group_name)
				{
					group_id = Exp_movie.groups[i].id;
					break;
				}
			}
			if (group_id < 0)
				group_id = Exp_movie.addGroup(group_name);

			// Create a new particle
			std::string part_name;
			part_name= integerToString( Exp_movie.particles.size() + 1); // start counting at 1
			part_id = Exp_movie.addParticle(part_name, my_random_subset);

			image_id = Exp_movie.addImage(group_id, mic_id, part_id);
			// Copy the current row of MDimgin into the current row of MDimg
			Exp_movie.MDimg.setObject(MDmovie.getObject(), image_id);

			// Set the orientations
			Exp_movie.MDimg.setValue(EMDL_ORIENT_ROT, rot, image_id);
			Exp_movie.MDimg.setValue(EMDL_ORIENT_TILT, tilt, image_id);
			Exp_movie.MDimg.setValue(EMDL_ORIENT_PSI, psi, image_id);
			Exp_movie.MDimg.setValue(EMDL_ORIENT_ORIGIN_X, xoff, image_id);
			Exp_movie.MDimg.setValue(EMDL_ORIENT_ORIGIN_Y, yoff, image_id);
			// Now also set the priors on the orientations equal to the orientations from the averages
			Exp_movie.MDimg.setValue(EMDL_ORIENT_ROT_PRIOR, rot, image_id);
			Exp_movie.MDimg.setValue(EMDL_ORIENT_TILT_PRIOR, tilt, image_id);
			Exp_movie.MDimg.setValue(EMDL_ORIENT_PSI_PRIOR, psi, image_id);
			Exp_movie.MDimg.setValue(EMDL_ORIENT_ORIGIN_X_PRIOR, xoff, image_id);
			Exp_movie.MDimg.setValue(EMDL_ORIENT_ORIGIN_Y_PRIOR, yoff, image_id);
			Exp_movie.MDimg.setValue(EMDL_PARTICLE_CLASS, my_class, image_id);
			Exp_movie.MDimg.setValue(EMDL_PARTICLE_RANDOM_SUBSET, my_random_subset, image_id);
			// Set normcorrection to 1
			double norm = 1.;
			Exp_movie.MDimg.setValue(EMDL_IMAGE_NORM_CORRECTION, norm);

			// Get the rlnParticleName and set this into rlnOriginalParticleName to prevent re-reading of this file to be handled differently..
			FileName fn_ori_part;
			Exp_movie.MDimg.getValue(EMDL_PARTICLE_NAME, fn_ori_part, image_id);
			Exp_movie.MDimg.setValue(EMDL_PARTICLE_ORI_NAME, fn_ori_part, image_id);
			// Set the particle number in its new rlnParticleName
			Exp_movie.MDimg.setValue(EMDL_PARTICLE_NAME, part_name, image_id);
			// Set the new group name
			Exp_movie.MDimg.setValue(EMDL_MLMODEL_GROUP_NAME, group_name);


			// Add ExpOriParticles
			// If this ori_particle did not exist in the Exp_movie yet, add it to the Exp_movie experiment
			long int ori_part_id = -1;
			for (long int i = 0; i < Exp_movie.ori_particles.size(); i++)
			{
				if (Exp_movie.ori_particles[i].name == fn_ori_part)
				{
					ori_part_id = i;
					break;
				}
			}
			// If no ExpOriParticles with this name was found, then add new one
			if (ori_part_id < 0)
				ori_part_id = Exp_movie.addOriginalParticle(fn_ori_part, my_random_subset);
			// Add this particle to the OriginalParticle
			// get Number from mic_name (-1 if empty mic_name, or no @ in mic_name)
			std::string fnt;
			long int my_order;
			mic_name.decompose(my_order, fnt);
			(Exp_movie.ori_particles[ori_part_id]).addParticle(part_id, my_random_subset, my_order);

		}
	}

	if (Exp_movie.MDimg.numberOfObjects() == 0)
		REPORT_ERROR("Experiment::expandToMovieFrames: ERROR: no movie frames selected. Check filenames of micrographs, movies and particle stacks!");

	// Now that all particles from MDmovie have been parsed, set nr_frames per particle in the metadatatable
	FOR_ALL_OBJECTS_IN_METADATA_TABLE(Exp_movie.MDimg)
	{
		Exp_movie.MDimg.setValue(EMDL_PARTICLE_NR_FRAMES, count_frames[(pointer_current_idx[current_object])]);
	}

	// Now replace the current Experiment with Exp_movie
	(*this) = Exp_movie;

	// Order the particles in each ori_particle
	orderParticlesInOriginalParticles();


}
void Experiment::orderParticlesInOriginalParticles()
{
	// If the orders are negative (-1) then dont sort anything
	if (ori_particles[0].particles_order[0] < 0)
		return;

	for (long int i = 0; i < ori_particles.size(); i++)
	{
		int nframe = ori_particles[i].particles_order.size();

		std::vector<std::pair<long int, long int> > vp;
        vp.reserve(nframe);
        for (long int j = 0; j < nframe; j++)
        	vp.push_back(std::make_pair(ori_particles[i].particles_order[j], j));
        // Sort on the first elements of the pairs
        std::sort(vp.begin(), vp.end());

        // tmp copy of particles_id
        std::vector<long int> _particles_id = ori_particles[i].particles_id;
        for (int j = 0; j < nframe; j++)
			ori_particles[i].particles_id[j] = _particles_id[vp[j].second];

		// We now no longer need the particles_order vector, clear it to save memory
		ori_particles[i].particles_order.clear();
	}

}

void Experiment::getAverageMicrographs()
{

	// Loop over all ori_particles and group in identical AverageMicrographs
	// Recognize identical AverageMicrographs, by common MicrographNames AFTER the "@" sign
	// This will only work for movie-processing, where the rlnMicrographName indeed contains an "@"

	average_micrographs.clear();
	for (long int i = 0; i < ori_particles.size(); i++)
	{
		// Get the first particle (i.e. movie-frame) from this original_particle (i.e. movie)
		long int part_id = ori_particles[i].particles_id[0];
		// The micrograph_id from the first image of the first particle (movie-particles only have 1 image/particle!!!!)
		// This whole series-stuff is not turning out to be extremely useful yet....
		long int mic_id = getMicrographId(part_id, 0);
		FileName mic_name = micrographs[mic_id].name;
		std::string avg_mic_name;
		long int frame_nr;
		// Get the part AFTER the "@" sign
		mic_name.decompose(frame_nr, avg_mic_name);
		// If this micrograph did not exist yet, add it to the experiment
		long int avg_mic_id = -1;
		for (long int ii = average_micrographs.size() - 1; ii >= 0; ii--) // search backwards to find match faster
		{
			if (average_micrographs[ii].name == avg_mic_name)
			{
				avg_mic_id = micrographs[ii].id;
				break;
			}
		}
		if (avg_mic_id < 0)
		{
			avg_mic_id = addAverageMicrograph(avg_mic_name);
		}

		// Add this OriginalParticle to the AverageMicrograph
		average_micrographs[avg_mic_id].ori_particles_id.push_back(i);
	}

}

void Experiment::usage()
{
	std::cout
	<< "  -i                     : Starfile with input images\n"
	;
}

// Read from file
void Experiment::read(FileName fn_exp, bool do_ignore_particle_name)
{

//#define DEBUG_READ
#ifdef DEBUG_READ
	std::cerr << "Entering Experiment::read" << std::endl;
	char c;
#endif

	// Initialize by emptying everything
	clear();
	MetaDataTable MDmicin, MDimgin;
	long int group_id, mic_id, part_id, image_id;

	if (!fn_exp.isStarFile())
	{
		// Read images from stack. Ignore all metadata, just use filenames
		// Add a single Micrograph
		group_id = addGroup("group");
		mic_id = addMicrograph("micrograph");

		// Check that a MRC stack ends in .mrcs, not .mrc (which will be read as a MRC 3D map!)
		if (fn_exp.contains(".mrc") && !fn_exp.contains(".mrcs"))
			REPORT_ERROR("Experiment::read: ERROR: MRC stacks of 2D images should be have extension .mrcs, not .mrc!");

		// Read in header-only information to get the NSIZE of the stack
		Image<double> img;
		img.read(fn_exp, false); // false means skip data, only read header

		for (long int n = 0; n <  NSIZE(img()); n++)
		{
			FileName fn_img;
			fn_img.compose(n+1, fn_exp); // fn_img = integerToString(n) + "@" + fn_exp;
			// Add the particle to my_area = 0
			part_id = addParticle("particle");
			// Add this image to the area
			image_id = addImage(group_id, mic_id, part_id);
			// Also add OriginalParticle
			(ori_particles[addOriginalParticle("particle")]).addParticle(part_id, 0, -1);

			// Set the filename and other metadata parameters
			MDimg.setValue(EMDL_IMAGE_NAME, fn_img, image_id);
		}

	}
	else
	{
		// Read all metadata from a STAR file
		bool contains_images_block;

		// First try reading a data_images block into MDimgin (as written by Experiment::write() )
		MDimgin.read(fn_exp, "images");
		// If that did not work, try reading the first data-block in the file
		if (MDimgin.isEmpty())
		{
			MDimgin.read(fn_exp);
			contains_images_block = false;
		}
		else
		{
			contains_images_block = true;
		}

#ifdef DEBUG_READ
	std::cerr << "Done reading MDimgin" << std::endl;
	std::cerr << "Press any key to continue..." << std::endl;
	std::cin >> c;
#endif

		// If there is no EMDL_MICROGRAPH_NAME, then just use a single group and micrograph
		if (!MDimgin.containsLabel(EMDL_MICROGRAPH_NAME))
		{
			group_id = addGroup("group");
			mic_id = addMicrograph("micrograph");
		}

		// Now Loop over all objects in the metadata file and fill the logical tree of the experiment
#ifdef DEBUG_READ
		std::cerr << " sizeof(int)= " << sizeof(int) << std::endl;
		std::cerr << " sizeof(long int)= " << sizeof(long int) << std::endl;
		std::cerr << " sizeof(double)= " << sizeof(double) << std::endl;
		std::cerr << " sizeof(ExpImage)= " << sizeof(ExpImage) << std::endl;
		std::cerr << " sizeof(ExpParticle)= " << sizeof(ExpParticle) << std::endl;
		std::cerr << " sizeof(ExpMicrograph)= " << sizeof(ExpMicrograph) << std::endl;
		std::cerr << " sizeof(std::vector<long int>)= " << sizeof(std::vector<long int>) << std::endl;
		long int nr_read = 0;

#endif
		// Reserve the same number of particles as there are images in MDimgin
		particles.reserve(MDimgin.size() * (sizeof(ExpParticle) + sizeof(ExpImage)));
		// TODO precalculate this somehow?! ARE THESE RESERVES NECESSARY ANYWAY???!!!
		micrographs.reserve(4000);

#ifdef DEBUG_READ
		std::cerr << "Done reserving" << std::endl;
		std::cerr << "Press any key to continue..." << std::endl;
		std::cin >> c;
#endif

		FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDimgin)
		{

			// Add new micrographs or get mic_id for existing micrograph
			FileName mic_name="", group_name="";
			if (MDimgin.containsLabel(EMDL_MICROGRAPH_NAME))
			{
				MDimgin.getValue(EMDL_MICROGRAPH_NAME, mic_name);

				// If this micrograph did not exist yet, add it to the experiment
				mic_id = -1;
				for (long int i = micrographs.size() - 1; i >= 0; i--) // search backwards to find match faster
				{
					if (micrographs[i].name == mic_name)
					{
						mic_id = micrographs[i].id;
						break;
					}
				}
				if (mic_id < 0)
					mic_id = addMicrograph(mic_name);

				// Check whether there is a group label, if not use a group for each micrograph
				if (MDimgin.containsLabel(EMDL_MLMODEL_GROUP_NAME))
				{
					MDimgin.getValue(EMDL_MLMODEL_GROUP_NAME, group_name);
				}
				else
				{
					group_name = mic_name;
				}
				// If this group did not exist yet, add it to the experiment
				group_id = -1;
				for (long int i = groups.size() - 1; i >= 0; i--) // search backwards to find match faster
				{
					if (groups[i].name == group_name)
					{
						group_id = groups[i].id;
						break;
					}
				}
				if (group_id < 0)
					group_id = addGroup(group_name);
			}
			else
			{
				// All images belong to the same micrograph
				mic_id = 0;
				group_id = 0;
			}

			// If there is an EMDL_PARTICLE_RANDOM_SUBSET entry in the input STAR-file, then set the random_subset, otherwise use defualt (0)
			int my_random_subset;
			if (!MDimgin.getValue(EMDL_PARTICLE_RANDOM_SUBSET, my_random_subset))
				my_random_subset = 0;

			// Add this image to an existing particle, or create a new particle
			std::string part_name;
			if (MDimgin.containsLabel(EMDL_PARTICLE_NAME) && !do_ignore_particle_name)
			{
				MDimgin.getValue(EMDL_PARTICLE_NAME, part_name);
				// Check whether this particle already exists
				part_id = -1;
				for (long int i = particles.size() - 1; i >= 0; i--) // search backwards to find match faster
				{
					if (particles[i].name == part_name)
					{
						part_id = particles[i].id;
						break;
					}
				}
				// If no particle with this name was found, then add new one
				if (part_id < 0)
				{
					part_id = addParticle(part_name, my_random_subset);
				}
			}
			else
			{
				// If particleName is not in the input metadata file, call this particle by the image number
				part_name= integerToString(particles.size());
				part_id = addParticle(part_name, my_random_subset);
			}

			// Add this particle to an existing OriginalParticle, or create a new OriginalParticle
			long int ori_part_id = -1;
			if (MDimgin.containsLabel(EMDL_PARTICLE_ORI_NAME))
			{
				MDimgin.getValue(EMDL_PARTICLE_ORI_NAME, part_name);
				for (long int i = ori_particles.size() - 1; i >= 0; i--)  // search backwards to find match faster
				{
					if (ori_particles[i].name == part_name)
					{
						ori_part_id = i;
						break;
					}
				}

				// If no OriginalParticles with this name was found, then add new one
				if (ori_part_id < 0)
					ori_part_id = addOriginalParticle(part_name, my_random_subset);

			}
			else
			{
				// If there are no EMDL_PARTICLE_ORI_NAME in the input file: just one particle per OriginalParticle
				ori_part_id = addOriginalParticle(part_name, my_random_subset);
			}

			// Add this particle to the OriginalParticle
			std::string fnt;
			long int my_order;
			mic_name.decompose(my_order, fnt);
			(ori_particles[ori_part_id]).addParticle(part_id, my_random_subset, my_order);

			long int img_id = addImage(group_id, mic_id, part_id);
			// Copy the current row of MDimgin into the current row of MDimg
			MDimg.setObject(MDimgin.getObject(), img_id);

			// The group number is only set upon reading: it is not read from the STAR file itself,
			// there the only thing that matters is the order of the micrograph_names
			// Write igroup+1, to start numbering at one instead of at zero
			MDimg.setValue(EMDL_MLMODEL_GROUP_NO, group_id + 1, img_id);

#ifdef DEBUG_READ
			nr_read++;
#endif
		} // end loop over all objects in MDimgin

#ifdef DEBUG_READ
		std::cerr << " MDimg.lastObject()= " << MDimg.lastObject() << std::endl;
		std::cerr << " nr_read= " << nr_read << " particles.size()= " << particles.size() << " micrographs.size()= " << micrographs.size();
#endif

		// Now, if this file was created by this class then also fill MDmic:
		if (contains_images_block)
		{
			MDmicin.read(fn_exp, "micrographs");
			if (!MDmicin.isEmpty())
			{
				std::vector<std::string> found_mic_names;
				FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDmicin)
				{
					std::string mic_name;
					if (!MDmicin.getValue(EMDL_MICROGRAPH_NAME, mic_name))
						REPORT_ERROR("Experiment::read ERROR: data_micrographs block should contain micrographName labels!");
					found_mic_names.push_back(mic_name);
					bool found = false;
					for (long int i = 0; i < micrographs.size(); i++)
					{
						if (micrographs[i].name == mic_name)
						{
							// Copy entire row from MDmicin to MDmic (at line micrographs[i].id)
							MDmic.setObject(MDmicin.getObject(), micrographs[i].id);
							break;
						}
					}
				}

				// Also check whether all micrographs have been found....
				for (int i = 0; i < micrographs.size(); i++)
				{
					bool found = false;
					for (int j = 0; j < found_mic_names.size(); j++)
					{
						if (found_mic_names[j] == micrographs[i].name)
						{
							found = true;
							break;
						}
					}
					if (!found)
						REPORT_ERROR("Did not find the following micrograph in the data_micrographs table: " + micrographs[i].name);
				}

			}
		}
	}

#ifdef DEBUG_READ
	std::cerr << "Done filling MDimg" << std::endl;
	std::cerr << "Press any key to continue..." << std::endl;
	std::cin >> c;
#endif

	// Make sure some things are always set in the MDimg
	bool have_rot  = MDimgin.containsLabel(EMDL_ORIENT_ROT);
	bool have_tilt = MDimgin.containsLabel(EMDL_ORIENT_TILT);
	bool have_psi  = MDimgin.containsLabel(EMDL_ORIENT_PSI);
	bool have_xoff = MDimgin.containsLabel(EMDL_ORIENT_ORIGIN_X);
	bool have_yoff = MDimgin.containsLabel(EMDL_ORIENT_ORIGIN_Y);
	bool have_clas = MDimgin.containsLabel(EMDL_PARTICLE_CLASS);
	bool have_norm = MDimgin.containsLabel(EMDL_IMAGE_NORM_CORRECTION);
	FOR_ALL_OBJECTS_IN_METADATA_TABLE(MDimg)
	{
		double dzero=0., done=1.;
		int izero = 0;
		if (!have_rot)
			MDimg.setValue(EMDL_ORIENT_ROT, dzero);
		if (!have_tilt)
			MDimg.setValue(EMDL_ORIENT_TILT, dzero);
		if (!have_psi)
			MDimg.setValue(EMDL_ORIENT_PSI, dzero);
		if (!have_xoff)
			MDimg.setValue(EMDL_ORIENT_ORIGIN_X, dzero);
		if (!have_yoff)
			MDimg.setValue(EMDL_ORIENT_ORIGIN_Y, dzero);
		if (!have_clas)
			MDimg.setValue(EMDL_PARTICLE_CLASS, izero);
		if (!have_norm)
			MDimg.setValue(EMDL_IMAGE_NORM_CORRECTION, done);
	}

#ifdef DEBUG_READ
	std::cerr << "Done setting defaults MDimg" << std::endl;
	std::cerr << "Press any key to continue..." << std::endl;
	std::cin >> c;
#endif

	// Also set the image_size (use the last image for that, still in fn_img)
	FileName fn_img;
	Image<double> img;
	MDimg.getValue(EMDL_IMAGE_NAME, fn_img, MDimg.firstObject());
	if (fn_img != "")
	{
		img.read(fn_img, false); //false means read only header, skip real data
		int image_size = XSIZE(img());
		if (image_size != YSIZE(img()))
			REPORT_ERROR("Experiment::read: xsize != ysize: only squared images allowed");
			// Add a single object to MDexp
			MDexp.addObject();
		MDexp.setValue(EMDL_IMAGE_SIZE, image_size);
		if (ZSIZE(img()) > 1)
		{
			if (image_size != ZSIZE(img()))
				REPORT_ERROR("Experiment::read: xsize != zsize: only cubed images allowed");
			MDexp.setValue(EMDL_IMAGE_DIMENSIONALITY, 3);
		}
		else
		{
			MDexp.setValue(EMDL_IMAGE_DIMENSIONALITY, 2);
		}
	}
	else
	{
		REPORT_ERROR("There are no images read in: please check your input file...");
	}

	// Order the particles in each ori_particle (only useful for realignment of movie frames)
	orderParticlesInOriginalParticles();

#ifdef DEBUG_READ
	std::cerr << "Writing out debug_data.star" << std::endl;
	write("debug");
	exit(0);
#endif
}

// Write to file
void Experiment::write(FileName fn_root)
{

	std::ofstream  fh;
	FileName fn_tmp = fn_root+"_data.star";
    fh.open((fn_tmp).c_str(), std::ios::out);
    if (!fh)
        REPORT_ERROR( (std::string)"Experiment::write: Cannot write file: " + fn_tmp);

    // Always write MDimg
    MDimg.write(fh);

    // Only write MDmic if more than one micrograph is stored there...
    //if (MDmic.lastObject() > 1)
    //	MDmic.write(fh);

    // Only write MDexp if something is stored there...
    //if (!MDexp.isEmpty())
    //	MDexp.write(fh);

	fh.close();

}
