#==========================================================================
# Makefile
#==========================================================================
# @brief: A makefile the compiles and runs the simgnn program
#
# @desc: 1. Enter "make" to compile & execute the simgnn program
#        2. Enter "make clean" to clean up the directory (before submission)


# Extract Vivado HLS include path
VHLS_PATH := $(dir $(shell which vivado_hls))/..
VHLS_INC ?= ${VHLS_PATH}/include

COMMON_REPO := ../../..

# wide Memory Access Application
include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk

# simgnn Host Application
simgnn_host_SRCS=./simgnn_host.cpp ./utils.cpp $(xcl2_SRCS)
simgnn_host_HDRS=$(xcl2_HDRS)
simgnn_host_CXXFLAGS=-I./ $(xcl2_CXXFLAGS) $(opencl_CXXFLAGS) -I${VHLS_INC} -DK_CONST=3
simgnn_host_LDFLAGS=$(opencl_LDFLAGS)

# simgnn Kernels
simgnn_kernel_SRCS=./simgnn.cpp
simgnn_kernel_CLFLAGS=-I./ -k simgnn_kernel


EXES=simgnn_host
XCLBINS=simgnn_kernel

XOS=simgnn_kernel

simgnn_kernel_XOS=simgnn_kernel

# check
check_EXE=simgnn_host ./data
check_XCLBINS=simgnn_kernel

CHECKS=check

include $(COMMON_REPO)/utility/rules.mk

