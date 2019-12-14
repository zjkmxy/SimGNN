#include <string.h>
#include <math.h>
#include <assert.h>
#include "const.h"

#define NROW 1200
#define NCOL 1200
#define NRESULT (NROW * NCOL)

int load_ntn(
  float W[nFilters3][nFilters3][nTensorNeurons],
  float WB[nTensorNeurons][nFilters3*2],
  float bias[nTensorNeurons],
  const float weights[],
  const int idx_st
){
  int idx = idx_st;
  for(int i = 0; i < nFilters3; i ++){
    for(int j = 0; j < nFilters3; j ++){
      for(int k = 0; k < nTensorNeurons; k ++){
        W[i][j][k] = weights[idx ++];
      }
    }
  }
  for(int i = 0; i < nTensorNeurons; i ++){
    for(int j = 0; j < nFilters3 * 2; j ++){
      WB[i][j] = weights[idx ++];
    }
  }
  for(int i = 0; i < nTensorNeurons; i ++){
    bias[i] = weights[idx ++];
  }
  return idx;
}

template<int inDim, int outDim>
int load_dense(
  float W[outDim][inDim],
  float bias[outDim],
  const float weights[],
  const int idx_st
){
  int idx = idx_st;
  for(int i = 0; i < outDim; i ++){
    for(int j = 0; j < inDim; j ++){
      W[i][j] = weights[idx ++];
    }
  }
  for(int i = 0; i < outDim; i ++){
    bias[i] = weights[idx ++];
  }
  return idx;
}

void ntn_first(
  const float W[nFilters3][nFilters3][nTensorNeurons],
  const float WB[nTensorNeurons][nFilters3*2],
  const float bias[nTensorNeurons],
  const float din1[nFilters3],
  float score_w[nFilters3][nTensorNeurons],
  float score[nTensorNeurons]
){
  for(int k = 0; k < nTensorNeurons; k ++){
    #pragma HLS unroll
    score[k] = bias[k];
  }

  for(int j = 0; j < nFilters3; j ++){
    for(int k = 0; k < nTensorNeurons; k ++){
      #pragma HLS pipeline
      score_w[j][k] = din1[0] * W[0][j][k] + din1[1] * W[1][j][k] + din1[2] * W[2][j][k] + din1[3] * W[3][j][k] +
                      din1[4] * W[4][j][k] + din1[5] * W[5][j][k] + din1[6] * W[6][j][k] + din1[7] * W[7][j][k] +
                      din1[8] * W[8][j][k] + din1[9] * W[9][j][k] + din1[10] * W[10][j][k] + din1[11] * W[11][j][k] +
                      din1[12] * W[12][j][k] + din1[13] * W[13][j][k] + din1[14] * W[14][j][k] + din1[15] * W[15][j][k];
    }
  }
  for(int k = 0; k < nTensorNeurons; k ++){
    #pragma HLS unroll
    for(int i = 0; i < nFilters3; i ++){
      #pragma HLS unroll
      score[k] += WB[k][i] * din1[i];
    }
  }
}

void ntn_next(
  const float WB[nTensorNeurons][nFilters3*2],
  const float din2[nFilters3],
  const float score_w[nFilters3][nTensorNeurons],
  const float score_0[nTensorNeurons],
  float dout[nTensorNeurons]
){
  float score[nTensorNeurons];
  #pragma HLS array_partition variable=score complete dim=1

  for(int k = 0; k < nTensorNeurons; k ++){
    #pragma HLS unroll
    score[k] = score_0[k];
  }

  for(int j = 0; j < nFilters3; j ++){
    #pragma HLS unroll
    for(int k = 0; k < nTensorNeurons; k ++){
      #pragma HLS unroll
      score[k] += score_w[j][k] * din2[j];
    }
  }

  for(int k = 0; k < nTensorNeurons; k ++){
    #pragma HLS unroll
    for(int i = 0; i < nFilters3; i ++){
      #pragma HLS unroll
      score[k] += WB[k][nFilters3 + i] * din2[i];
    }
  }

  for(int k = 0; k < nTensorNeurons; k ++){
    #pragma HLS unroll
    dout[k] = fmax(0.f, score[k]);
  }
}

void l1_calc(
  const float W[nDense1][nScores],
  const float bias[nDense1],
  const float din[nScores],
  float dout[nDense1]
){
  for(int k = 0; k < nDense1; k ++){
    #pragma HLS unroll
    dout[k] = bias[k];
  }
  for(int i = 0; i < nScores; i ++){
    #pragma HLS unroll
    for(int k = 0; k < nDense1; k ++){
      #pragma HLS unroll
      dout[k] += W[k][i] * din[i];
    }
  }
  for(int k = 0; k < nDense1; k ++){
    #pragma HLS unroll
    dout[k] = fmax(0, dout[k]);
  }
}

void l2_calc(
  const float W[nDense2][nDense1],
  const float bias[nDense2],
  const float din[nDense1],
  float dout[nDense2]
){
  for(int k = 0; k < nDense2; k ++){
    #pragma HLS unroll
    dout[k] = bias[k];
  }
  for(int k = 0; k < nDense2; k ++){
    #pragma HLS unroll
    for(int i = 0; i < nDense1; i ++){
      #pragma HLS unroll
      dout[k] += W[k][i] * din[i];
    }
  }
  for(int k = 0; k < nDense2; k ++){
    #pragma HLS unroll
    dout[k] = fmax(0, dout[k]);
  }
}

void l3_calc(
  const float W[nDense3][nDense2],
  const float bias[nDense3],
  const float din[nDense2],
  float dout[nDense3]
){
  dout[0] = bias[0];
  for(int i = 0; i < nDense2; i ++){
    #pragma HLS unroll
    dout[0] += W[0][i] * din[i];
  }
}

float process_pair(
  const float NTN_WB[nTensorNeurons][nFilters3*2],
  const float score_w[nFilters3][nTensorNeurons],
  const float score_0[nTensorNeurons],
  const float L1_W[nDense1][nScores],
  const float L1_bias[nDense1],
  const float L2_W[nDense2][nDense1],
  const float L2_bias[nDense2],
  const float L3_W[nDense3][nDense2],
  const float L3_bias[nDense3],
  const float col_emb[SIZE_EMBEDDING]
){
  float r[SIZE_EMBEDDING];
  float ntn_ret[nTensorNeurons];
  float l1_ret[nDense1], l2_ret[nDense2], l3_ret[nDense3];
  #pragma HLS array_partition variable=r complete dim=1
  #pragma HLS array_partition variable=ntn_ret complete dim=1
  #pragma HLS array_partition variable=l1_ret complete dim=1
  #pragma HLS array_partition variable=l2_ret complete dim=1
  #pragma HLS array_partition variable=l3_ret complete dim=1

  for(int i = 0; i < SIZE_EMBEDDING; i ++){
    #pragma HLS unroll
    r[i] = col_emb[i];
  }

  ntn_next(NTN_WB, r, score_w, score_0, ntn_ret);
  l1_calc(L1_W, L1_bias, ntn_ret, l1_ret);
  l2_calc(L2_W, L2_bias, l1_ret, l2_ret);
  l3_calc(L3_W, L3_bias, l2_ret, l3_ret);
  return 1.f / (1.f + exp(- l3_ret[0]));
}

extern "C" {

#pragma ACCEL kernel
void simgnn_kernel(
  const float *weights,
  const int nrow,
  const float *row_embs,
  const int ncol,
  const float *col_embs,
  float *results
){

#pragma HLS INTERFACE m_axi port=col_embs offset=slave depth=19200
#pragma HLS INTERFACE m_axi port=results offset=slave depth=1440000
#pragma HLS INTERFACE m_axi port=row_embs offset=slave depth=19200
#pragma HLS INTERFACE m_axi port=weights offset=slave depth=4801
#pragma HLS INTERFACE s_axilite port=col_embs bundle=control
#pragma HLS INTERFACE s_axilite port=ncol bundle=control
#pragma HLS INTERFACE s_axilite port=nrow bundle=control
#pragma HLS INTERFACE s_axilite port=results bundle=control
#pragma HLS INTERFACE s_axilite port=row_embs bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  float NTN_W[nFilters3][nFilters3][nTensorNeurons];
  float NTN_WB[nTensorNeurons][nFilters3*2];
  float NTN_bias[nTensorNeurons];

  #pragma HLS array_partition variable=NTN_W complete dim=1
  #pragma HLS array_partition variable=NTN_W cyclic dim=3 factor=2
  #pragma HLS array_partition variable=NTN_WB complete dim=1
  #pragma HLS array_partition variable=NTN_WB complete dim=2
  #pragma HLS array_partition variable=NTN_bias complete dim=1

  float L1_W[nDense1][nScores];
  float L1_bias[nDense1];
  float L2_W[nDense2][nDense1];
  float L2_bias[nDense2];
  float L3_W[nDense3][nDense2];
  float L3_bias[nDense3];

  #pragma HLS array_partition variable=L1_W complete dim=2
  #pragma HLS array_partition variable=L1_W complete dim=1
  #pragma HLS array_partition variable=L1_bias complete dim=1
  #pragma HLS array_partition variable=L2_W complete dim=2
  #pragma HLS array_partition variable=L2_W complete dim=1
  #pragma HLS array_partition variable=L2_bias complete dim=1
  #pragma HLS array_partition variable=L3_W complete dim=2
  #pragma HLS array_partition variable=L3_W complete dim=1
  #pragma HLS array_partition variable=L3_bias complete dim=1
  
  float re[1200][16], ce[1200][16];
  float result_cache[1200];
  #pragma HLS array_partition variable=re complete dim=2
  #pragma HLS array_partition variable=re cyclic dim=1 factor=4
  #pragma HLS array_partition variable=ce complete dim=2
  #pragma HLS array_partition variable=ce cyclic dim=1 factor=4
  #pragma HLS array_partition variable=result_cache cyclic dim=1 factor=4

  float score_w[nFilters3][nTensorNeurons];
  float score_0[nTensorNeurons];
  #pragma HLS array_partition variable=score_w complete dim=1
  #pragma HLS array_partition variable=score_w complete dim=2
  #pragma HLS array_partition variable=score_0 complete dim=1

  assert(nrow == NROW);
  assert(ncol == NCOL);

  int idx = 0;
  idx = load_ntn(NTN_W, NTN_WB, NTN_bias, weights, idx);
  idx = load_dense<nScores, nDense1>(L1_W, L1_bias, weights, idx);
  idx = load_dense<nDense1, nDense2>(L2_W, L2_bias, weights, idx);
  idx = load_dense<nDense2, nDense3>(L3_W, L3_bias, weights, idx);

  for(int row = 0; row < nrow; row ++){
    #pragma HLS LOOP_TRIPCOUNT max=1200 min=1200
    for(int j = 0; j < 16; j ++){
      re[row][j] = row_embs[row*16+j];
    }
  }
  for(int col = 0; col < ncol; col ++){
    #pragma HLS LOOP_TRIPCOUNT max=1200 min=1200
    for(int j = 0; j < 16; j ++){
      ce[col][j] = col_embs[col*16+j];
    }
  }

  for(int row = 0; row < nrow; row ++){
    #pragma HLS LOOP_TRIPCOUNT max=1200 min=1200

    ntn_first(NTN_W, NTN_WB, NTN_bias, re[row], score_w, score_0);
    for(int col = 0; col < ncol; col ++){
      #pragma HLS pipeline
      #pragma HLS LOOP_TRIPCOUNT max=1200 min=1200
      result_cache[col] = process_pair(NTN_WB, score_w, score_0, L1_W, L1_bias, L2_W, L2_bias, L3_W, L3_bias, ce[col]);
      results[row * 1200 + col] = result_cache[col];
    }
  }
}

} // extern "C"
