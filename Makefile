CUDA_DIR := /usr/local/cuda-7.5
BUILD_DIR :=bin
# CUDA architecture setting: going with all of them.
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
		-gencode arch=compute_20,code=sm_21 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64 $(CUDA_DIR)/lib
CXX = mpic++
NVCC = $(CUDA_DIR)/bin/nvcc
NVCCFLAGS := -ccbin $(CXX) -w

CC = g++ -w
CCDEPMODE = depmode=gcc3
CFLAGS = -g -O2
CPP = gcc -E -w
CXXCPP = g++ -E -w
CXXDEPMODE = depmode=gcc3
CXXFLAGS = -g -O2 -w
LIBS = -lfftw3_threads -lfftw3 -lpthread -ldl -lm -lX11 -lcudart -lcublas -lcurand -lcufft 
RELION_API_VERSION = 1.3
RELION_SO_VERSION = 1:3:0

INCDIR = -I./ -I./include
INCDIR += $(foreach includedir,$(CUDA_INCLUDE_DIR),-I$(includedir))

LDFLAGS = -L./lib

LDFLAGS += $(foreach librarydir,$(CUDA_LIB_DIR),-L$(librarydir)) 


SOURCES = src/ml_optimiser.cpp src/ml_optimiser_mpi.cpp src/parallel.cpp \
	src/ml_model.cpp src/exp_model.cpp src/fftw.cpp \
	src/metadata_table.cpp src/args.cpp src/mpi.cpp  \
	src/assembly.cpp src/error.cpp src/healpix_sampling.cpp \
	src/metadata_label.cpp src/strings.cpp src/symmetries.cpp \
	src/tabfuncs.cpp src/filename.cpp src/funcs.cpp src/time.cpp \
	src/backprojector.cpp src/mask.cpp src/complex.cpp src/euler.cpp \
	src/projector.cpp src/memory.cpp src/metadata_container.cpp \
	src/transformations.cpp src/matrix1d.cpp src/matrix2d.cpp \
	src/image.cpp src/numerical_recipes.cpp src/ctf.cpp

CU_SRCS := $(shell find src/ -name "*.cu")
CU_OBJS := $(addprefix $(BUILD_DIR)/, ${CU_SRCS:.cu=.cuo})
OBJS := $(addprefix $(BUILD_DIR)/, ${SOURCES:.cpp=.o})

OBJ_BUILD_DIR := $(BUILD_DIR)/src
APP_BUILD_DIR := $(BUILD_DIR)/src/apps
Hea_BUILD_DIR := $(BUILD_DIR)/src/Healpix_2.15a

APP_ARCS := src/apps/refine.cpp  src/apps/refine_mpi.cpp 
refine_mpi_OBJ = $(APP_BUILD_DIR)/refine_mpi.o
refine_OBJ = $(APP_BUILD_DIR)/refine.o


ALL_BUILD_DIRS := $(BUILD_DIR) $(OBJ_BUILD_DIR) $(APP_BUILD_DIR) $(Hea_BUILD_DIR)

CC_OBJS = $(Hea_BUILD_DIR)/healpix_base.o
OBJS += $(CU_OBJS) $(CC_OBJS)

all: $(ALL_BUILD_DIRS) $(BUILD_DIR)/gerelion_refine $(BUILD_DIR)/gerelion_refine_mpi 
.PHONY: all

$(ALL_BUILD_DIRS):
	mkdir -p  $@

$(OBJ_BUILD_DIR)/%.o:src/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCDIR) $(LDFLAGS) $(CFLAGS)  -c $< -o  $@

$(Hea_BUILD_DIR)/healpix_base.o:src/Healpix_2.15a/healpix_base.cc
	$(NVCC) $(NVCCFLAGS) $(INCDIR) $(LDFLAGS) $(CFLAGS)  -c $< -o  $@

$(BUILD_DIR)/src/%.cuo: src/%.cu 
	$(NVCC) $(NVCCFLAGS) $(INCDIR) $(LDFLAGS) $(CFLAGS) $(CUDA_ARCH) -c $< -o  $@

$(APP_BUILD_DIR)/%.o:$(APP_ARCS)
	$(NVCC) $(NVCCFLAGS) $(INCDIR) $(LDFLAGS) $(CFLAGS)  -c $< -o  $@

$(BUILD_DIR)/gerelion_refine: $(refine_OBJ) $(OBJS) 
	$(CXX) -o $(BUILD_DIR)/gerelion_refine $(refine_OBJ) $(OBJS) $(LDFLAGS) $(CFLAGS) $(LIBS)

$(BUILD_DIR)/gerelion_refine_mpi: $(refine_mpi_OBJ) $(OBJS) 
	$(CXX)  -o $(BUILD_DIR)/gerelion_refine_mpi $(refine_mpi_OBJ) $(OBJS) $(LDFLAGS) $(CFLAGS) $(LIBS) 

.PHONY : clean
clean:
	rm -rf $(BUILD_DIR)
