#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstdio>
#include <xcl2.hpp>
#include "const.h"
#include "ml-layers.hpp"

#define NUM_ROW 1200
#define NUM_COL 1200

using std::vector;
using std::cout;
using std::endl;
using std::string;

float __attribute__((aligned(64))) row_embs[NUM_ROW][SIZE_EMBEDDING];
float __attribute__((aligned(64))) col_embs[NUM_COL][SIZE_EMBEDDING];
float __attribute__((aligned(64))) weights[nWeights];
float __attribute__((aligned(64))) results[NUM_ROW * NUM_COL];
int row_ids[NUM_ROW];
int col_ids[NUM_COL];

struct SingleGraph;
FILE* open_model(const char* dir);
SingleGraph* create_single(FILE* file);
void del_single(SingleGraph* p);
void read_ids(const char* dir, const char* name, int N, int ids[]);
void calc_embs(SingleGraph* calc, const char* dir, const char* name, float embs[][SIZE_EMBEDDING],
               const int ids[], int size);
void load_weights(FILE *f, float weights[]);

int main(int argc, char **argv) {
  FILE *fmodel = open_model(argv[1]);
  SingleGraph *calc;

  // Calculate embeddings
  calc = create_single(fmodel);
  read_ids(argv[1], "train", NUM_ROW, row_ids);
  read_ids(argv[1], "test", NUM_COL, col_ids);
  calc_embs(calc, argv[1], "train", row_embs, row_ids, NUM_ROW);
  calc_embs(calc, argv[1], "test", col_embs, col_ids, NUM_COL);
  del_single(calc);

  // Load weights
  load_weights(fmodel, weights);
  fclose(fmodel);

  // Evaluate scores
    // OpenCL host setup start
  vector<cl::Device> devices = xcl::get_xil_devices();
  cl::Device device = devices[0];

  cl::Context context(device);
  cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
  string device_name = device.getInfo<CL_DEVICE_NAME>(); 

  string binaryFile = xcl::find_binary_file(device_name, "simgnn_kernel");
  cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
  devices.resize(1);
  cl::Program program(context, devices, bins);
  cl::Kernel kernel(program, "simgnn_kernel");

  vector<cl::Memory> inBufVec, outBufVec;
  cl::Buffer buf_weight(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                        nWeights * sizeof(float), &weights[0]);
  cl::Buffer buf_row(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     NUM_ROW * SIZE_EMBEDDING * sizeof(float), &row_embs[0]);
  cl::Buffer buf_col(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     NUM_COL * SIZE_EMBEDDING * sizeof(float), &col_embs[0]);
  cl::Buffer buf_results(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
                         NUM_ROW * NUM_COL * sizeof(float), &results[0]);
  inBufVec.push_back(buf_weight);
  inBufVec.push_back(buf_row);
  inBufVec.push_back(buf_col);
  outBufVec.push_back(buf_results);

  //Copy input data to device global memory
  q.enqueueMigrateMemObjects(inBufVec, 0/* 0 means from host*/);

  auto krnl_simgnn = cl::KernelFunctor<cl::Buffer&, int, cl::Buffer&, int, cl::Buffer&, cl::Buffer&>(kernel);
  // OpenCL host setup end

  krnl_simgnn(cl::EnqueueArgs(q, cl::NDRange(1, 1, 1), cl::NDRange(1, 1, 1)),
              buf_weight, NUM_ROW, buf_row, NUM_COL, buf_col, buf_results);

  q.enqueueMigrateMemObjects(outBufVec, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();

  printf("RESULTS: %f %f %f %f\n", results[0], results[1], results[NUM_COL], results[NUM_COL+1]);

  return EXIT_SUCCESS;
}
