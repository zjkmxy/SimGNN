# Copyright (C) 2019 Falcon Computing Solutions, Inc. - All rights reserved.
#
# Choose target FPGA platform & vendor
VENDOR=XILINX
#DEVICE=xilinx_u250_xdma_201830_1
DEVICE=xilinx_aws-vu9p-f1-04261818_dynamic_5_0

# Edit Options per your setup if using AWS F1
S3_OPT=-s3_bucket=xinyuma-fpga-bucket -s3_dcp_key=tarballs -s3_logs_key=logs

# Host Code Compilation settings
SRC_DIR:=..
#INC_DIR:=/opt/Xilinx/Vivado/2019.1.op2552052/include
INC_DIR:=
#COMMON_DIR:=../../common
HOST_SRC_FILES=$(SRC_DIR)/simgnn_host.cpp $(SRC_DIR)/utils.cpp
DATA_DIR=../data
F1_DATA_DIR=../data

# Executable names and arguments
EXE=simgnn_host
ACC_EXE=$(EXE)_acc

# Testing mode
EXE_ARGS= ../data
F1_EXE_ARGS= ./data

CXX_INC_DIRS=-I. -I$(INC_DIR)
CXX_FLAGS+= $(CXX_INC_DIRS)
CXX_FLAGS+= -Wall -O3 -lrt
#CXX_FLAGS+= -g
ifeq ($(VENDOR),XILINX)
CXX_FLAGS +=-lstdc++ -L$(XILINX_SDX)/lib/lnx64.o
endif

# Accelerated Kernel settings
KERNEL_NAME=simgnn_kernel
KERNEL_SRC_FILES=$(SRC_DIR)/simgnn.cpp
KERNEL_INC_DIR=$(CXX_INC_DIRS)

# MerlinCC Options
#AUTODSE=1
CMP_OPT=-funsafe-math-optimizations --attribute memory_burst=on
LNK_OPT=

#common make
MCC_COMMON_DIR=./
include $(MCC_COMMON_DIR)/mcc_common.mk