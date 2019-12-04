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
  int idx
){
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
  int idx
){
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

void ntn_calc(
  const float W[nFilters3][nFilters3][nTensorNeurons],
  const float WB[nTensorNeurons][nFilters3*2],
  const float bias[nTensorNeurons],
  const float din1[nFilters3],
  const float din2[nFilters3],
  float dout[nTensorNeurons]
){
  float score_w[nFilters3][nTensorNeurons], score[nTensorNeurons];
  memset(score_w, 0, sizeof(score_w));
  for(int k = 0; k < nTensorNeurons; k ++){
    score[k] = bias[k];
  }
  for(int i = 0; i < nFilters3; i ++){
    for(int j = 0; j < nFilters3; j ++){
      for(int k = 0; k < nTensorNeurons; k ++){
        score_w[j][k] += din1[i] * W[i][j][k];
      }
    }
  }
  for(int j = 0; j < nFilters3; j ++){
    for(int k = 0; k < nTensorNeurons; k ++){
      score[k] += score_w[j][k] * din2[j];
    }
  }

  for(int k = 0; k < nTensorNeurons; k ++){
    for(int i = 0; i < nFilters3; i ++){
      score[k] += WB[k][i] * din1[i];
    }
    for(int i = 0; i < nFilters3; i ++){
      score[k] += WB[k][nFilters3 + i] * din2[i];
    }
  }

  for(int k = 0; k < nTensorNeurons; k ++){
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
    dout[k] = bias[k];
  }
  for(int k = 0; k < nDense1; k ++){
    for(int i = 0; i < nScores; i ++){
      dout[k] += W[k][i] * din[i];
    }
  }
  for(int k = 0; k < nDense1; k ++){
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
    dout[k] = bias[k];
  }
  for(int k = 0; k < nDense2; k ++){
    for(int i = 0; i < nDense1; i ++){
      dout[k] += W[k][i] * din[i];
    }
  }
  for(int k = 0; k < nDense2; k ++){
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
    dout[0] += W[0][i] * din[i];
  }
}

float process_pair(
  float NTN_W[nFilters3][nFilters3][nTensorNeurons],
  float NTN_WB[nTensorNeurons][nFilters3*2],
  float NTN_bias[nTensorNeurons],
  float L1_W[nDense1][nScores],
  float L1_bias[nDense1],
  float L2_W[nDense2][nDense1],
  float L2_bias[nDense2],
  float L3_W[nDense3][nDense2],
  float L3_bias[nDense3],
  const float row_emb[SIZE_EMBEDDING],
  const float col_emb[SIZE_EMBEDDING]
){
  float l[SIZE_EMBEDDING], r[SIZE_EMBEDDING];
  float ntn_ret[nTensorNeurons];
  float l1_ret[nDense1], l2_ret[nDense2], l3_ret[nDense3];

  for(int i = 0; i < SIZE_EMBEDDING; i ++){
    l[i] = row_emb[i];
  }
  for(int i = 0; i < SIZE_EMBEDDING; i ++){
    r[i] = col_emb[i];
  }

  ntn_calc(NTN_W, NTN_WB, NTN_bias, l, r, ntn_ret);
  l1_calc(L1_W, L1_bias, ntn_ret, l1_ret);
  l2_calc(L2_W, L2_bias, l1_ret, l2_ret);
  l3_calc(L3_W, L3_bias, l2_ret, l3_ret);
  return 1.f / (1.f + exp(- l3_ret[0]));
}

#pragma ACCEL kernel
void simgnn_kernel(
  const float weights[],
  const int nrow,
  const float row_embs[][SIZE_EMBEDDING],
  const int ncol,
  const float col_embs[][SIZE_EMBEDDING],
  float results[]
){
#pragma ACCEL interface variable=weights depth=4801
#pragma ACCEL interface variable=row_embs depth=1200
#pragma ACCEL interface variable=col_embs depth=1200
#pragma ACCEL interface variable=results depth=1440000
  float NTN_W[nFilters3][nFilters3][nTensorNeurons];
  float NTN_WB[nTensorNeurons][nFilters3*2];
  float NTN_bias[nTensorNeurons];

  float L1_W[nDense1][nScores];
  float L1_bias[nDense1];
  float L2_W[nDense2][nDense1];
  float L2_bias[nDense2];
  float L3_W[nDense3][nDense2];
  float L3_bias[nDense3];

  assert(nrow == NROW);
  assert(ncol == NCOL);

  int idx = 0;
  idx = load_ntn(NTN_W, NTN_WB, NTN_bias, weights, idx);
  idx = load_dense<nScores, nDense1>(L1_W, L1_bias, weights, idx);
  idx = load_dense<nDense1, nDense2>(L2_W, L2_bias, weights, idx);
  idx = load_dense<nDense2, nDense3>(L3_W, L3_bias, weights, idx);

  for(int row = 0; row < nrow; row ++){
    for(int col = 0; col < ncol; col ++){
      results[row*NCOL+col] = process_pair(NTN_W, NTN_WB, NTN_bias, L1_W, L1_bias, L2_W, L2_bias, L3_W, L3_bias, row_embs[row], col_embs[col]);
    }
  }
}
