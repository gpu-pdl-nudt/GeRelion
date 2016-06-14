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


#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <climits>
#include <algorithm>
#include <vector>
#include <typeinfo>

#include "src/numerical_recipes.h"
#include "src/macros.h"
#include "src/error.h"
#include "src/transformations.h"

//Including the CUDA lib related header files
#include <cublas_v2.h>  //The CUDA blas library
#include <cufft.h>

#define BLOCK_SIZE 256
#define BLOCK_SIZE_128 128

#define BLOCK_X 32
#define BLOCK_Y 8

#define NEAREST_NEIGHBOUR 0
#define TRILINEAR 1
#define CONVOLUTE_BLOB 2

__device__ double atomicAdd(double* address, double val);
extern "C"
{
	void relion_gpu_scal(const int N,  double* alpha, double* X, int stride);
}
template<typename T>
void centerFFT_gpu(T* in, T* out, int nr_images, int dim, int xdim, int ydim, int zdim, bool forward);

template <typename T>
void selfTranslate_gpu(T* V1,
                       double*  Aref_matrxi,
                       double* exp_old_offset_D,
                       int dim, int  xdim, int ydim, int image_size,  int nr_image,
                       bool wrap = WRAP, T outside = 0);
template <typename T>
void    do_norm_correction_gpu(T* image_D, double* normcorr_D, int image_size, int exp_nr_images, double avg_norm_correction);
template<typename T>
void translate_gpu(T* V1,
                   T* V2,
                   const  double*  v,
                   int dim,
                   bool wrap = WRAP, T outside = 0);

template <typename T>
void calculate_local_sqrtXi2_gpu(T* local_Fimgs_D, double* exp_local_sqrtXi2_D, int nr_images, int image_size);

void calculate_Minvsigma2_gpu(double* exp_Minvsigma2_D,
                              int* local_myMresol_D,
                              double* sigma2_noise_D,
                              int* group_id_D,
                              double sigma2_fudge,
                              int nr_images,
                              int image_size,
                              int myMresol_size,
                              int noise_size_of_group);
void calculate_frefctf_Mctf_gpu(cufftDoubleComplex* frefctf_D,
                                cufftDoubleComplex* fref_D,
                                double* mctf_D,
                                double* mctf_out_D,
                                double* myscale_D,
                                int nr_orients,
                                int nr_oversampled_rot,
                                int nr_ipart,
                                int image_size,
                                bool do_ctf_correction_and_refs_are_ctf_corrected,
                                bool do_scale_correction);

void  calculate_frefctf_gpu(cufftDoubleComplex* frefctf_D,
                            cufftDoubleComplex* fref_D, double* exp_local_Fctfs_D,
                            double* myscale_D,
                            int nr_images,
                            int nr_orients,
                            int nr_oversampled_rot,
                            int image_size,
                            bool do_ctf_correction_and_refs_are_ctf_corrected,
                            bool do_scale_correction);

void calculate_wdiff2_sumXA_Meta_total_gpu(
    cufftDoubleComplex* Frecctf_D,
    cufftDoubleComplex* Fimg_shift_D,
    double* weight_D,
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
    double* thr_wsum_sigma2_noise_D,
    double* thr_wsum_norm_correction_D,
    double* thr_wsum_scale_correction_XA_D,
    double* thr_wsum_scale_correction_AA_D,
    double* data_vs_prior_class_D,
    int* mresol_fine_D,
    int* group_id_D,
    int thr_wsum_size,
    int ref_dim,
    int modelorientational_prior_mode,
    int exp_nr_psi,
    double model_prior_offset_class_x,
    double model_prior_offset_class_y,
    double* exp_old_offset_D,
    double* oversampled_translations_D,
    double* pointer_dir_nonzeroprior_D,
    double* thr_sumw_group_D,
    double* thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D,
    double* exp_prior_D,
    double* thr_wsum_pdf_direction_D
);

void calculate_wdiff2_sumXA_Meta_total_OPT_gpu(
    cufftDoubleComplex* Frecctf_D,
    cufftDoubleComplex* Fimg_shift_D,
    double* weight_D,
    int* weight_index,
    double* exp_Weight_D,
    double* exp_Sum_Weigh_particles,
    int* Significant_list_D,
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
    double* thr_wsum_sigma2_noise_D,
    double* thr_wsum_norm_correction_D,
    double* thr_wsum_scale_correction_XA_D,
    double* thr_wsum_scale_correction_AA_D,
    double* data_vs_prior_class_D,
    int* mresol_fine_D,
    int* group_id_D,
    int thr_wsum_size,
    int ref_dim,
    int modelorientational_prior_mode,
    int exp_nr_psi,
    double model_prior_offset_class_x,
    double model_prior_offset_class_y,
    double* exp_old_offset_D,
    double* oversampled_translations_D,
    double* pointer_dir_nonzeroprior_D,
    double* thr_sumw_group_D,
    double* thr_wsum_pdf_class_sigma2_offset_prior_offsetx_D,
    double* exp_prior_D,
    double* thr_wsum_pdf_direction_D,
    int weight_size_x
);

void calculate_sum_shift_img_gpu(cufftDoubleComplex* Fimg_D,
                                 double* fweight_D,
                                 cufftDoubleComplex* Fimg_shift_nomask_D,
                                 double* Minvsigma2_D,
                                 double* weight_D,
                                 double* mctf_D,
                                 int* weight_index,
                                 int*    isSignificant_D,
                                 int image_size,
                                 int nr_weight,
                                 int nr_images,
                                 int nr_orients,
                                 int nr_oversampled_rot,
                                 int nr_trans,
                                 int nr_oversampled_trans,
                                 int nr_valid_orients);

void calculate_sum_shift_img_shared_gpu(cufftDoubleComplex* Fimg_D,
                                        double* fweight_D,
                                        cufftDoubleComplex* Fimg_shift_nomask_D,
                                        double* Minvsigma2_D,
                                        double* weight_D,
                                        double* mctf_D,
                                        double* exp_Sum_Weigh_particles,
                                        int* Significant_list_D,
                                        int image_size,
                                        int nr_images,
                                        int nr_oversampled_rot,
                                        int nr_trans,
                                        int nr_oversampled_trans,
                                        int nr_valid_orients,
                                        int exp_Mweight_xdim
                                       );


void calculate_exp_mweight_gpu(double* exp_Mweight_D,
                               double* exp_min_diff2_D,
                               double* diff2_D,
                               bool* do_proceed_D,
                               bool* is_last_image_in_series_D,
                               int iorientclass,
                               int exp_nr_particles,
                               int exp_nr_trans,
                               int exp_nr_oversampled_rot,
                               int exp_nr_oversampled_trans,
                               int exp_Mweight_D_size,
                               int exp_iseries);



void calculate_diff2_do_squared_difference_gpu(double* diff2_D,
                                               cufftDoubleComplex* frefctf_D,
                                               cufftDoubleComplex* Fimg_shift_D,
                                               double* exp_highres_Xi2_imgs_D,
                                               double* Minvsigma2_D,
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
                                               int nr_valid_orient_trans);

void calculate_diff2_no_do_squared_difference_gpu(double* diff2_D,
                                                  cufftDoubleComplex* frefctf_D,
                                                  cufftDoubleComplex* Fimg_shift_D,
                                                  double* exp_local_sqrtXi2_D,
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
                                                  int nr_valid_orient_trans);
void init_exp_mweight_gpu(double* exp_Mweight_D, double c, int size);
void init_exp_min_diff2_gpu(double* exp_min_diff2_D, double c, int size);


void get2DFourierTransform_gpu(cufftDoubleComplex* Fref_all_D,
                               double* A_D,
                               cufftDoubleComplex* data_D,
                               bool inv, int padding_factor,
                               int r_max, int r_min_nn,
                               int f2d_x,
                               int f2d_y,
                               int data_x,
                               int data_y,
                               int data_z,
                               int data_starty,
                               int data_startz,
                               int nr_A,
                               int ref_dim);

void backproject_gpu(cufftDoubleComplex* f2d_D,
                     double* A_D,
                     bool inv,
                     double* Mweight_D,
                     cufftDoubleComplex* data_D,
                     double* weight_D,
                     double padding_factor,
                     int r_max,
                     int r_min_nn,
                     int ydim_f2d,
                     int xdim_f2d,
                     int nr_valid_orients,
                     int nr_oversampled_rot,
                     int image_size,
                     int interpolator,
                     int start_y,
                     int start_z,
                     int data_YXDIM,
                     int data_XDIM,
                     int ref_dim
                    );
void  sign_Weight_gpu(double* exp_Mweight_D,
                      double* exp_Sum_Weigh_particles,
                      int* isSignificant_D,
                      int* valid_weight_list_D,
                      int exp_Mweight_xdim,
                      int nr_images,
                      int nr_orients,
                      int nr_oversampled_rot,
                      int nr_trans,
                      int nr_oversampled_trans);

void compact_Weight_gpu(double* exp_Mweight_D,
                        double* exp_Sum_Weigh_particles,
                        int* valid_weight_list_D,
                        int* compact_position_list_D,
                        int exp_Mweight_xdim,
                        int nr_images,
                        int nr_orients,
                        int nr_oversampled_rot,
                        int nr_trans,
                        int nr_oversampled_trans,
                        int iorientclass_offset,
                        double* valid_weight_D,
                        //int * ipart_id_D,
                        //int *ishift_id_D,
                        // int *exp_orient_id_D,
                        // int *exp_rot_id_D,
                        int* weight_index,
                        int* image_index_flag_D);



void re_initial_exp_weight_gpu(double* exp_Mweight_D,
                               bool* exp_Mcoarse_significant_D,
                               double* mini_weight_particel_D,
                               int nr_images,
                               int exp_Mweight_D_size,
                               int exp_ipass);


void calculate_weight_gpu(const int nr_particles,
                          double* exp_sum_weight_D,
                          int* exp_none_zero_number,
                          double* max_weight,
                          double* exp_Mweight_D,
                          double* exp_min_diff2_D,
                          double* pdf_orientation_D,
                          double* pdf_offset_D,
                          long int xdim_Mweight,
                          int iclass_min,
                          int iclass_max,
                          int model_nr_classes,
                          int exp_nr_classes,
                          long int nr_elements,
                          long int nr_orients,
                          long int exp_nr_trans,
                          long int exp_nr_oversampled_rot,
                          long int exp_nr_oversampled_trans);

void calculate_weight_first_iter_gpu(double* exp_Mweight_D,
                                     bool* exp_Mcoarse_significant_D,
                                     double* exp_min_diff2_D,
                                     const int nr_particles,
                                     long int xdim_Mweight,
                                     int iclass_min,
                                     int iclass_max,
                                     long int nr_elements,
                                     int exp_ipass);

void calculate_minimal_weight_per_particle_gpu(double* exp_Mweight_D,
                                               double* Mini_weight,
                                               double init_value,
                                               int valid_size,
                                               int nr_particles,
                                               int exp_Mweight_D_size);
void calculate_sum_weight_per_particle_gpu(double* exp_Mweight_D,
                                           double* sum_weight,
                                           int* local_none_zero_number,
                                           int* total_none_zero_number,
                                           double* max_weight_per_particle_D,
                                           double* max_weight_D,
                                           int valid_size,
                                           int nr_particles,
                                           int exp_Mweight_D_size);
void calculate_significant_partial_sum_gpu(double* exp_Mweight_D,
                                           double* thresh_weight_D,
                                           double* sum_weight,
                                           int valid_size,
                                           int nr_particles,
                                           int exp_Mweight_D_size,
                                           int down_factor);

void calculate_sum_significant_weight_per_particle_gpu(double* exp_Mweight_D,
                                                       double* sum_weight,
                                                       int valid_size,
                                                       int nr_particles,
                                                       int exp_Mweight_D_size);
void calculate_diff2_do_squared_difference_pass0_gpu(double* diff2_D,
                                                     cufftDoubleComplex* frefctf_D,
                                                     cufftDoubleComplex* Fimg_shift_D,
                                                     double* exp_highres_Xi2_imgs_D,
                                                     double* Minvsigma2_D,
                                                     int* valid_orient_trans_index_D,
                                                     int exp_nr_particles,
                                                     int exp_nr_orients,
                                                     int exp_nr_trans,
                                                     int exp_nr_oversampled_rot,
                                                     int exp_nr_oversampled_trans,
                                                     int nr_valid_orients,
                                                     int diff_xdim,
                                                     int image_size
                                                    );
void calculate_diff2_no_do_squared_difference_pass0_gpu(double* diff2_dev,
                                                        cufftDoubleComplex* Frefctf_dev,
                                                        cufftDoubleComplex* Fimg_shift_dev,
                                                        double* exp_local_sqrtXi2_dev,
                                                        int* valid_orient_trans_index_D,
                                                        int exp_nr_particles,
                                                        int exp_nr_orients,
                                                        int exp_nr_trans,
                                                        int exp_nr_oversampled_rot,
                                                        int exp_nr_oversampled_trans,
                                                        int nr_valid_orients,
                                                        int diff_xdim,
                                                        int image_size);

void apply_ctf_and_calculate_all_diff2_squared_pass0_gpu(cufftDoubleComplex* fref_D,
                                                         double* exp_local_Fctfs_D,
                                                         double* myscale_D,
                                                         double* diff2_D,
                                                         cufftDoubleComplex* Fimg_shift_D,
                                                         double* exp_highres_Xi2_imgs_D,
                                                         double* Minvsigma2_D,
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
                                                         bool do_scale_correction);

void calculate_img_power_gpu(cufftDoubleComplex* local_Faux_D,
                             double* exp_power_imgs_D,
                             double* exp_highres_Xi2_imgs_D,
                             int nr_images,
                             int Faux_x,
                             int Faux_y,
                             int Faux_z,
                             int mymodel_ori_size,
                             int mymodel_current_size
                            );
void getFftwImage_gpu(double* local_Fctf_images_D,
                      double* ctf_related_parameters_H,
                      int orixdim, int oriydim, double angpix,
                      bool do_abs, bool do_only_flip_phases, bool do_intact_until_first_peak,
                      bool do_damping,
                      int nr_images,
                      int xdim,
                      int ydim,
                      bool do_ctf_correction);

void ScaleComplexPointwise_gpu(cufftDoubleComplex* fFourier_D, int size, double scale);
#endif

