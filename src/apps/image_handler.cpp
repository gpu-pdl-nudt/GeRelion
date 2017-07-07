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

#include <src/image.h>
#include <src/funcs.h>
#include <src/args.h>
#include <src/fftw.h>
#include <src/time.h>
#include <src/symmetries.h>

class image_handler_parameters
{
	public:
   	FileName fn_in, fn_out, fn_sel, fn_img, fn_sym, fn_sub, fn_mult, fn_div, fn_add, fn_subtract, fn_mtf;
	int bin_avg, avg_first, avg_last, edge_x0, edge_xF, edge_y0, edge_yF, filter_edge_width, new_box;
	bool do_add_edge, do_flipXY, do_flipmXY, do_shiftCOM;
	double multiply_constant, divide_constant, add_constant, subtract_constant, threshold_above, threshold_below, angpix, new_angpix, lowpass, highpass, bfactor, shift_x, shift_y, shift_z;
   	// I/O Parser
	IOParser parser;

	Image<double> Iout;

	void usage()
	{
		parser.writeUsage(std::cerr);
	}

	void read(int argc, char **argv)
	{

		parser.setCommandLine(argc, argv);

		int general_section = parser.addSection("General options");
	    fn_in = parser.getOption("--i", "Input image (.mrc) or movie/stack (.mrcs)");
	    fn_out = parser.getOption("--o", "Output file");

	    int cst_section = parser.addSection("image-by-constant operations");
	    multiply_constant = textToFloat(parser.getOption("--multiply_constant", "Multiply the image(s) pixel values by this constant", "1"));
	    divide_constant = textToFloat(parser.getOption("--divide_constant", "Divide the image(s) pixel values by this constant", "1"));
	    add_constant = textToFloat(parser.getOption("--add_constant", "Add this constant to the image(s) pixel values", "0."));
	    subtract_constant = textToFloat(parser.getOption("--subtract_constant", "Subtract this constant from the image(s) pixel values", "0."));
	    threshold_above = textToFloat(parser.getOption("--threshold_above", "Set all values higher than this value to this value", "999."));
	    threshold_below = textToFloat(parser.getOption("--threshold_below", "Set all values lower than this value to this value", "-999."));

	    int img_section = parser.addSection("image-by-image operations");
	    fn_mult = parser.getOption("--multiply", "Multiply input image(s) by the pixel values in this image", "");
	    fn_div = parser.getOption("--divide", "Divide input image(s) by the pixel values in this image", "");
	    fn_add = parser.getOption("--add", "Add the pixel values in this image to the input image(s) ", "");
	    fn_subtract = parser.getOption("--subtract", "Subtract the pixel values in this image to the input image(s) ", "");

	    int four_section = parser.addSection("per-image operations");
	    fn_mtf = parser.getOption("--correct_mtf", "STAR-file with MTF values to correct for (rlnResolutionInversePixel and rlnMtfValue)", "");
	    bfactor = textToFloat(parser.getOption("--bfactor", "Apply a B-factor (in A^2)", "0."));
	    lowpass = textToFloat(parser.getOption("--lowpass", "Low-pass filter frequency (in A)", "-1."));
	    highpass = textToFloat(parser.getOption("--highpass", "High-pass filter frequency (in A)", "-1."));
	    angpix = textToFloat(parser.getOption("--angpix", "Pixel size (in A)", "1."));
	    new_angpix = textToFloat(parser.getOption("--rescale_angpix", "Scale input image(s) to this new pixel size (in A)", "-1."));
	    new_box = textToInteger(parser.getOption("--new_box", "Resize the image(s) to this new box size (in pixel) ", "-1"));
	    filter_edge_width = textToInteger(parser.getOption("--filter_edge_width", "Width of the raised cosine on the low/high-pass filter edge (in resolution shells)", "2"));
	    do_shiftCOM = parser.checkOption("--shift_com", "Shift image(s) to their center-of-mass (only on positive pixel values)");
	    shift_x = textToFloat(parser.getOption("--shift_x", "Shift images this many pixels in the X-direction", "0."));
	    shift_y = textToFloat(parser.getOption("--shift_y", "Shift images this many pixels in the Y-direction", "0."));
	    shift_z = textToFloat(parser.getOption("--shift_z", "Shift images this many pixels in the Z-direction", "0."));

	    int three_d_section = parser.addSection("3D operations");
	    fn_sym = parser.getOption("--sym", "Symmetrise 3D map with this point group (e.g. D6)", "");

	    int preprocess_section = parser.addSection("2D-micrograph (or movie) operations");
	    do_flipXY = parser.checkOption("--flipXY", "Flip the image(s) in the XY direction?");
	    do_flipmXY = parser.checkOption("--flipmXY", "Flip the image(s) image(s)in the -XY direction?");
	    do_add_edge = parser.checkOption("--add_edge", "Add a barcode-like edge to the micrograph/movie frames?");
	    edge_x0 = textToInteger(parser.getOption("--edge_x0", "Pixel column to be used for the left edge", "0"));
	    edge_y0 = textToInteger(parser.getOption("--edge_y0", "Pixel row to be used for the top edge", "0"));
	    edge_xF = textToInteger(parser.getOption("--edge_xF", "Pixel column to be used for the right edge", "4095"));
	    edge_yF = textToInteger(parser.getOption("--edge_yF", "Pixel row to be used for the bottom edge", "4095"));

	    int avg_section = parser.addSection("Movie-frame averaging options");
       	bin_avg = textToInteger(parser.getOption("--avg_bin", "Width (in frames) for binning average, i.e. of every so-many frames", "-1"));
    	avg_first = textToInteger(parser.getOption("--avg_first", "First frame to include in averaging", "-1"));
    	avg_last = textToInteger(parser.getOption("--avg_last", "Last frame to include in averaging", "-1"));

    	// Check for errors in the command-line option
    	if (parser.checkForErrors())
    		REPORT_ERROR("Errors encountered on the command line (see above), exiting...");

	}

	void run()
	{

		Image<double> Ihead, Iin;
		MultidimArray<double> Mtmp;

		Ihead.read(fn_in, false);
		int xdim, ydim, zdim;
		long int ndim;
		Ihead.getDimensions(xdim, ydim, zdim, ndim);

		if (zdim > 1 && (do_add_edge || do_flipXY || do_flipmXY))
			REPORT_ERROR("ERROR: you cannot perform 2D operations like --add_edge, --flipXY or --flipmXY on 3D maps. If you intended to operate on a movie, use .mrcs extensions for stacks!");

		if (zdim > 1 && (bin_avg > 0 || (avg_first >= 0 && avg_last >= 0)))
			REPORT_ERROR("ERROR: you cannot perform movie-averaging operations on 3D maps. If you intended to operate on a movie, use .mrcs extensions for stacks!");

		int avgndim = 1;
		if (bin_avg > 0)
		{
			avgndim = ndim / bin_avg;
		}
		Image<double> Iavg(xdim, ydim, zdim, avgndim);
		Image<double> Iout(xdim, ydim, zdim);
		Image<double> Iop;
		if (fn_mult != "")
			Iop.read(fn_mult);
		else if (fn_div != "")
			Iop.read(fn_div);
		else if (fn_add != "")
			Iop.read(fn_add);
		else if (fn_subtract != "")
			Iop.read(fn_subtract);

		if (fn_mult != "" || fn_div != "" || fn_add != "" || fn_subtract != "")
			if (XSIZE(Iop()) != xdim || YSIZE(Iop()) != ydim || ZSIZE(Iop()) != zdim)
				REPORT_ERROR("Error: operate-image is not of the correct size");

		// Read each frame and do whatever is asked
		time_config();
   		init_progress_bar(ndim);
		bool do_rewrite;
		for (long int nn = 0; nn < ndim; nn++)
		{
			if (ndim > 1)
				Iin.read(fn_in, true, nn);
			else
				Iin.read(fn_in);

			do_rewrite = false;

			// 2D options
			if (do_add_edge)
			{
				do_rewrite = true;
				// Treat X-boundaries
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iin())
				{
					if (j < edge_x0)
						DIRECT_A2D_ELEM(Iin(), i, j) = DIRECT_A2D_ELEM(Iin(), i, edge_x0);
					else if (j > edge_xF)
						DIRECT_A2D_ELEM(Iin(), i, j) = DIRECT_A2D_ELEM(Iin(), i, edge_xF);
				}
				// Treat Y-boundaries
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iin())
				{
					if (i < edge_y0)
						DIRECT_A2D_ELEM(Iin(), i, j) = DIRECT_A2D_ELEM(Iin(), edge_y0, j);
					else if (i > edge_yF)
						DIRECT_A2D_ELEM(Iin(), i, j) = DIRECT_A2D_ELEM(Iin(), edge_yF, j);
				}
			}
			// Flipping: this needs to be done from Iin to Iout (i.e. can't be done on-line on Iout only!)
			if (do_flipXY)
			{
				do_rewrite = true;
				// Flip X/Y
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iin())
				{
					DIRECT_A2D_ELEM(Iout(), i, j) = DIRECT_A2D_ELEM(Iin(), j, i);

				}
			}
			else if (do_flipmXY)
			{
				do_rewrite = true;
				// Flip mX/Y
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iin())
				{
					DIRECT_A2D_ELEM(Iout(), i, j) = DIRECT_A2D_ELEM(Iin(), XSIZE(Iin()) - 1 - j, YSIZE(Iin()) - 1 - i);
				}
			}
			else
			{
				Iout = Iin;
			}


			// From here on also 3D options
			if (fabs(multiply_constant - 1.) > 0.)
			{
				do_rewrite = true;
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Iin())
				{
					DIRECT_A3D_ELEM(Iout(), k, i, j) *= multiply_constant;
				}
			}
			else if (fabs(divide_constant - 1.) > 0.)
			{
				do_rewrite = true;
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Iin())
				{
					DIRECT_A3D_ELEM(Iout(), k, i, j) /= divide_constant;
				}
			}
			else if (fabs(add_constant) > 0.)
			{
				do_rewrite = true;
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Iin())
				{
					DIRECT_A3D_ELEM(Iout(), k, i, j) += add_constant;
				}
			}
			else if (fabs(subtract_constant) > 0.)
			{
				do_rewrite = true;
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Iin())
				{
					DIRECT_A3D_ELEM(Iout(), k, i, j) -= subtract_constant;
				}
			}
			else if (fn_mult != "")
			{
				do_rewrite = true;
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Iin())
				{
					DIRECT_A3D_ELEM(Iout(), k, i, j) *= DIRECT_A3D_ELEM(Iop(), k, i, j);
				}
			}
			else if (fn_div != "")
			{
				do_rewrite = true;
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Iin())
				{
					if (DIRECT_A3D_ELEM(Iop(), k, i, j) < 1e-10)
						std::cout << "Warning: very small pixel values in divide image..." << std::endl;
					DIRECT_A3D_ELEM(Iout(), k, i, j) /= DIRECT_A3D_ELEM(Iop(), k, i, j);
				}
			}
			else if (fn_add != "")
			{
				do_rewrite = true;
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Iin())
				{
					DIRECT_A3D_ELEM(Iout(), k, i, j) += DIRECT_A3D_ELEM(Iop(), k, i, j);
				}
			}
			else if (fn_subtract != "")
			{
				do_rewrite = true;
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Iin())
				{
					DIRECT_A3D_ELEM(Iout(), k, i, j) -= DIRECT_A3D_ELEM(Iop(), k, i, j);
				}
			}

			// More 2D/3D stuff
			if (fn_mtf != "")
			{
				do_rewrite = true;
				correctMapForMTF(Iout(), fn_mtf);
			}
			if (fabs(bfactor) > 0.)
			{
				do_rewrite = true;
				applyBFactorToMap(Iout(), bfactor, angpix);
			}
			if (lowpass > 0.)
			{
				do_rewrite = true;
				lowPassFilterMap(Iout(), lowpass, angpix, filter_edge_width);
			}
			if (highpass > 0.)
			{
				do_rewrite = true;
				highPassFilterMap(Iout(), lowpass, angpix, filter_edge_width);
			}

			// Shifting
			if (do_shiftCOM)
			{
				do_rewrite = true;
				selfTranslateCenterOfMassToCenter(Iout(), DONT_WRAP);
			}
			else if (fabs(shift_x) > 0. || fabs(shift_y) > 0. || fabs(shift_z) > 0.)
			{
				do_rewrite = true;
				Matrix1D<double> shift(2);
				XX(shift) = shift_x;
				YY(shift) = shift_y;
				if (zdim > 1)
				{
					shift.resize(3);
					ZZ(shift) = shift_z;
				}
				selfTranslate(Iout(), shift, DONT_WRAP);
			}

			// Re-scale
			if (new_angpix > 0.)
			{
				do_rewrite = true;

				int oldsize = XSIZE(Iout());
				int newsize = ROUND(oldsize * (angpix / new_angpix));
				resizeMap(Iout(), newsize);

				// Also reset the sampling rate in the header
				Iout.MDMainHeader.setValue(EMDL_IMAGE_SAMPLINGRATE_X, new_angpix);
				Iout.MDMainHeader.setValue(EMDL_IMAGE_SAMPLINGRATE_Y, new_angpix);
				if (Iout().getDim() == 3)
					Iout.MDMainHeader.setValue(EMDL_IMAGE_SAMPLINGRATE_Z, new_angpix);

			}

			// Re-window
			if (new_box > 0)
			{
				do_rewrite = true;
				Iout().setXmippOrigin();
				if (Iout().getDim() == 2)
				{
					Iout().window(FIRST_XMIPP_INDEX(new_box), FIRST_XMIPP_INDEX(new_box),
							   LAST_XMIPP_INDEX(new_box),  LAST_XMIPP_INDEX(new_box));
				}
				else if (Iout().getDim() == 3)
				{
					Iout().window(FIRST_XMIPP_INDEX(new_box), FIRST_XMIPP_INDEX(new_box), FIRST_XMIPP_INDEX(new_box),
							   LAST_XMIPP_INDEX(new_box),  LAST_XMIPP_INDEX(new_box),  LAST_XMIPP_INDEX(new_box));
				}
			}

			// 3D-only stuff
			if (fn_sym != "")
			{
				do_rewrite = true;
				symmetriseMap(Iout(), fn_sym);
			}

			// Thresholding (can be done after any other operation)
			if (fabs(threshold_above - 999.) > 0.)
			{
				do_rewrite = true;
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Iin())
				{
					if (DIRECT_A3D_ELEM(Iout(), k, i, j) > threshold_above)
						DIRECT_A3D_ELEM(Iout(), k, i, j) = threshold_above;
				}
			}
			if (fabs(threshold_below + 999.) > 0.)
			{
				do_rewrite = true;
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Iin())
				{
					if (DIRECT_A3D_ELEM(Iout(), k, i, j) < threshold_below)
						DIRECT_A3D_ELEM(Iout(), k, i, j) = threshold_below;
				}
			}

			if (do_rewrite)
			{
				if (ndim > 1)
				{
					// Movies: first frame overwrite, then append
					if (nn == 0)
						Iout.write(fn_out, -1, (ndim > 1), WRITE_OVERWRITE);
					else
						Iout.write(fn_out, -1, false, WRITE_APPEND);
				}
				else
				{
					// Check whether fn_out has an "@": if so REPLACE the corresponding frame in the output stack!
					long int n;
					FileName fn_tmp;
					fn_out.decompose(n,fn_tmp);
					n--;
					if (n >= 0)
						Iout.write(fn_tmp, n, true, WRITE_REPLACE);
					else
						Iout.write(fn_out);
				}
			}

			// Take care of averaging (this again is done in 2D!)
			if (bin_avg > 0)
			{
				int myframe = nn / bin_avg;
				if (myframe < avgndim)
				{
					FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iout())
					{
						DIRECT_NZYX_ELEM(Iavg(),myframe,0,i,j) += DIRECT_A2D_ELEM(Iout(), i, j); // just store sum
					}
				}
			}
			else if (avg_first >= 0 && avg_last >= 0 && nn+1 >= avg_first && nn+1 <= avg_last) // add one to start counting at 1
			{
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Iout())
				{
					DIRECT_MULTIDIM_ELEM(Iavg(), n) += DIRECT_MULTIDIM_ELEM(Iout(), n); // just store sum
				}
			}

			progress_bar(nn);
		}
   		progress_bar(ndim);

   		if (bin_avg > 0 || (avg_first >= 0 && avg_last >= 0))
   		{
   			Iavg.write(fn_out);
   			std::cout << " Done! Written output as: " << fn_out << std::endl;
   		}
   		else if (do_rewrite)
   		{
   			std::cout << " Done! Written output as: " << fn_out << std::endl;
   		}
   		else
   		{
   			std::cout << " Done nothing, please provide an operation to perform ... " << std::endl;
   		}


	}


};


int main(int argc, char *argv[])
{
	image_handler_parameters prm;

	try
    {

		prm.read(argc, argv);

		prm.run();

    }
    catch (RelionError XE)
    {
        prm.usage();
        std::cout << XE;
        exit(1);
    }
    return 0;
}



