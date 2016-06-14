# GeRelion

GeRelion is a GPU-enhanced parallel implementation of single particle cryo-EM image processing.
Developed by Huayou Su, Wen Wen, Xiaoli Du and Dongsheng Li from University of Defense Technology. 
For the current version, we implementation 
the most time-comsuming components of RELION with GPU, including 3D classificaion and auto-refine.

The implementation is built upon RELION (REgularised LIkelihood Optimisation).
RELION is a stand-alone computer program that employs an empirical Bayesian approach to refinement of (multiple) 3D reconstructions or 2D class averages in electron cryo-microscopy (cryo-EM). It is developed in the group of Sjors Scheres at the MRC Laboratory of Molecular Biology.
[More detailed information about RELION can be found](http://www2.mrc-lmb.cam.ac.uk/relion/index.php/Main_Page)


## Prerequisites
----------------
1. Hardware requirements
To use the program GeRelion, you should have the CUDA-enabled GPUs from NVIDIA Corporation.
GeRelion prefers NVIDIA GPUs with arch_sm equals 3.5 or prior.
2. Software requirements
The CUDA Toolkit must be installed successfully. In addition,
the thrust library and [MPI](https://www.open-mpi.org/) pragraming interface should be installed. 

 
## Compiling the GeRelion
-------------------------
1. Change the directory of your project in Makefile, check line 1.
CUDA_DIR := path to your cuda directory installed (such as /usr/local/cuda-7.5) 

2. Move to your GeRelion project folder and run make to build the program gerelion_refine and gereline_refine_mpi.
We recommend to use parallel compilation "make -j n" (n is the number of thread to compiled the program).
The compilation will last for several minites due to the usage of thrust library. 
The executable binary file will be put in the build directory of the project. 

## Running GeRelion
-------------------
The current version only support the command line mode. The basic parameter of GeRelion
is almost the same of Relion except of adding one parameter "--mode". 
Parameter "--mode" indicates using the CPU mode or GPU mode. 0 is for CPU, 1 is for GPU mode.
You can download the [TRPV1 dataset](https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10005/)

The following command is a example of runnning GeRelion on two nodes(node01 and node02), each one with 4 K40 GPUs, GERELION_HOME is the directory of your GeRelion project:
mpirun --np 9 -N 5 --host node01,node02 $GERELION_HOME/build/gereline_refine_mpi --o Class3D_OPT/run8 --i particles_autopick_sort_class2d.star --particle_diameter 200 --angpix 3.54 --ref 3i3e_lp50A.mrc --firstiter_cc --ini_high 50 --ctf --ctf_corrected_ref --iter 25 --tau2_fudge 2 --K 4 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --offset_range 3 --offset_step 2 --sym C1 --norm --scale  --j 1 --memory_per_thread 4 --dont_combine_weights_via_disc --mode 1



