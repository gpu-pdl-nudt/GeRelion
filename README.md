# GeRelion

GeRelion is a GPU-enhanced parallel implementation of single particle cryo-EM image processing.
Developed by Huayou Su, Wen Wen, Xiaoli Du, Xicheng Lu,Dongsheng Li from National University of Defense Technology and Maofu Liao from Harvard Medical School. 
For the current version, we implemented 
the most time-comsuming components of RELION with GPU, including _3D classificaion_ and _auto-refine_.

The implementation is built upon RELION (REgularised LIkelihood Optimisation).
RELION is a stand-alone computer program that employs an empirical Bayesian approach to refinement of (multiple) 3D reconstructions or 2D class averages in electron cryo-microscopy (cryo-EM). It is developed in the group of Sjors Scheres at the MRC Laboratory of Molecular Biology.
[More detailed information about RELION can be found](http://www2.mrc-lmb.cam.ac.uk/relion/index.php/Main_Page)

The original RELION codes for data read/write and MPI communication are unmodified, 
and the flow control of progressive processing in the original RELION is kept. 
The _*.cu_ source files  and their corresponding header files in directory _src_ are developed by GeRelion team.
The _*.cpp_ source files  and their corresponding header files in directory _src_ are from relion-1.3.
Some of the _*.ccp_ and _*.hpp_ files from relion are modified by GeRelion team.

## Prerequisites
----------------
1. Hardware requirements
To use the program GeRelion, you should have the CUDA-enabled GPUs from NVIDIA Corporation.
GeRelion prefers NVIDIA GPUs with arch_sm equals 3.5 or prior.
2. Software requirements
The CUDA Toolkit must be installed successfully. In addition,
the thrust library and [MPI](https://www.open-mpi.org/) pragraming interface should be installed. 
3. The thiry party library fftw and fltk is required, both of them are put int the folder _external_.
 
## Compiling the GeRelion
-------------------------
1. Change the directory of your project in Makefile, check line 1.
   CUDA_DIR := path to your cuda directory installed (such as /usr/local/cuda-7.5) 

2. Move to your GeRelion project folder and run the script _INSTALL.sh_.
The compilation will last for several minites due to the usage of thrust library. 
The executable binary file will be put in  _bin_ directory of the project. 

## Running GeRelion
-------------------
Before running the program, you may need to export the environment variable *_LD_LIBRARY_PATH_* to 
your lib directory, such as *export* *LD_LIBRARY_PAYH=$GERELION_HOME/lib:$LD_LIBRARY_PATH*
The current version only support the command line mode. The basic parameter of GeRelion
is almost the same of Relion except of adding one parameter _--mode_. 
Parameter "--mode" indicates using the CPU mode or GPU mode. 0 is for CPU, 1 is for GPU mode.
You can download the [TRPV1 dataset](https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10005/)

The following command is a example of runnning GeRelion on two nodes(node01 and node02), each one with 4 K40 GPUs, GERELION_HOME is the directory of your GeRelion project:

mpirun --np 9 --host node01,node02 $GERELION_HOME/bin/gereline_refine_mpi --o Class3D_OPT/run8 --i particles_autopick_sort_class2d.star --particle_diameter 200 --angpix 3.54 --ref 3i3e_lp50A.mrc --firstiter_cc --ini_high 50 --ctf --ctf_corrected_ref --iter 25 --tau2_fudge 2 --K 4 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --offset_range 3 --offset_step 2 --sym C1 --norm --scale  --j 1 --memory_per_thread 4 --dont_combine_weights_via_disc --mode 1


## License
----------
GeRelion is released under the terms of the GNU [General Public License](https://opensource.org/licenses/gpl-license) as published by the Free Software Foundation.


Contact:
Huayou Su
huayousu@163.com