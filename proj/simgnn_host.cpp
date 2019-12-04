#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstdio>
#include "const.h"
#include "ml-layers.hpp"
#ifdef MCC_ACC
#include MCC_ACC_H_FILE
#else
extern void simgnn_kernel(
  const float weights[],
  const int nrow,
  const float row_embs[][SIZE_EMBEDDING],
  const int ncol,
  const float col_embs[][SIZE_EMBEDDING],
  float results[]
);
#endif

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

#ifdef MCC_ACC // use kernel binary file
  char* kernel_bin_file = argv[argc - 1];
  // load the binary file into the system
  __merlin_init(kernel_bin_file);
#endif

#ifdef MCC_ACC
  __merlin_simgnn_kernel(weights, NUM_ROW, row_embs, NUM_COL, col_embs, results);
#else
  simgnn_kernel(weights, NUM_ROW, row_embs, NUM_COL, col_embs, results);
#endif

#ifdef MCC_ACC
    __merlin_release();
#endif

  printf("RESULTS: %f %f %f %f\n", results[0], results[1], results[NUM_COL], results[NUM_COL+1]);

  return EXIT_SUCCESS;
}
