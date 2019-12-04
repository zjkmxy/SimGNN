#ifndef _ML_LAYERS_H
#define _ML_LAYERS_H

#include <cstdio>
#include <cmath>
#include <cstring>
#include "const.h"

typedef int (*GRAPH)[2];

inline float sigmoid(float x){
  return 1.f / (1.f + exp(- x));
}

template<int B>
inline void relu(int A, float M[][B]){
  for(int i = 0; i < A; i ++){
    for(int j = 0; j < B; j ++){
      M[i][j] = fmax(0.f, M[i][j]);
    }
  }
}

inline void relu(int N, float F[]){
  for(int i = 0; i < N; i ++){
    F[i] = fmax(0.f, F[i]);
  }
}

template<int nIn, int nOut>
struct GCNConv{
  float W[nIn][nOut];
  float bias[nOut];

  void load(FILE *f){
    fread(&W, sizeof(W), 1, f);
    fread(&bias, sizeof(bias), 1, f);
  }

  // N = #nodes, M = #edges (including self), G[M][2] (directed), din[N][nIn], dOut[N][nOut]
  void calc(int N, int M, const int G[][2], const float din[][nIn], float dout[][nOut]){
    float deg[N], degsqrt[N], norm[M];
    float AX[N][nIn];
    memset(deg, 0, sizeof(deg));
    memset(AX, 0, sizeof(AX));
    for(int j = 0; j < M; j ++){
      deg[G[j][0]] += 1;
    }
    for(int i = 0; i < N; i ++){
      degsqrt[i] = sqrt(deg[i]);
    }
    for(int j = 0; j < M; j ++){
      norm[j] = 1.f / (degsqrt[G[j][0]] * degsqrt[G[j][1]]);
    }
    for(int j = 0; j < M; j ++){
      int st = G[j][0], ed = G[j][1];
      for(int f = 0; f < nIn; f ++){
        AX[ed][f] += norm[j] * din[st][f];
      }
    }
    for(int i = 0; i < N; i ++){
      for(int g = 0; g < nOut; g ++){
        dout[i][g] = bias[g];
        for(int f = 0; f < nIn; f ++){
          dout[i][g] += AX[i][f] * W[f][g];
        }
        // dout[i][g] = max(0, dout[i][g]);
      }
    }
  }
};

template<int F>
struct Attention{
  float W[F][F];
  void load(FILE* f){
    fread(&W, sizeof(W), 1, f);
  }
  
  void calc(int N, const float din[][F], float dout[F]){
    float global_context[F], sig_scores[N];
    memset(global_context, 0, sizeof(global_context));
    memset(sig_scores, 0, sizeof(sig_scores));
    for(int i = 0; i < N; i ++){
      for(int f = 0; f < F; f ++){
        for(int g = 0; g < F; g ++){
          global_context[g] += din[i][f] * W[f][g];
        }
      }
    }
    for(int f = 0; f < F; f ++){
      global_context[f] = tanh(global_context[f] / N);
    }
    
    for(int i = 0; i < N; i ++){
      for(int f = 0; f < F; f ++){
        sig_scores[i] += din[i][f] * global_context[f];
      }
    }
    for(int i = 0; i < N; i ++){
      sig_scores[i] = sigmoid(sig_scores[i]);
    }
    
    for(int f = 0; f < F; f ++){
      dout[f] = 0;
    }
    for(int i = 0; i < N; i ++){
      for(int f = 0; f < F; f ++){
        dout[f] += din[i][f] * sig_scores[i];
      }
    }
  }
};

template<int F>
struct Average{
  void calc(int N, const float din[][F], float dout[F]){
    for(int k = 0; k < F; k ++){
      dout[k] = 0.0f;
    }
    for(int i = 0; i < N; i ++){
      for(int k = 0; k < F; k ++){
        dout[k] += din[i][k];
      }
    }
    for(int k = 0; k < F; k ++){
      dout[k] /= N;
    }
  }
};

template<int F, int Bin>
struct Histogram{
  void calc(int N1, int N2, const float din1[][F], const float din2[][F], float dout[Bin]){
    float scores[N1][N2];
    float minv = 1e30, maxv = 0, dist;
    for(int i = 0; i < N1; i ++){
      for(int j = 0; j < N2; j ++){
        scores[i][j] = 0;
        for(int k = 0; k < F; k ++){
          scores[i][j] += din1[i][k] * din2[j][k];
        }
        minv = fmin(minv, scores[i][j]);
        maxv = fmax(maxv, scores[i][j]);
      }
    }
    for(int k = 0; k < Bin; k ++){
      dout[k] = 0;
    }
    dist = (maxv - minv) / Bin;
    for(int i = 0; i < N1; i ++){
      for(int j = 0; j < N2; j ++){
        int pos = floor((scores[i][j] - minv) / dist);
        pos = fmax(fmin(pos, Bin - 1), 0);
        dout[pos] += 1;
      }
    }
    for(int k = 0; k < Bin; k ++){
      dout[k] /= 1.0f * N1 * N2;
    }
  }
};

template<int nIn, int nOut>
struct TensorNet{
  float W[nIn][nIn][nOut];
  float WB[nOut][nIn*2];
  float bias[nOut];

  void load(FILE* f){
    fread(&W, sizeof(W), 1, f);
    fread(&WB, sizeof(WB), 1, f);
    fread(&bias, sizeof(bias), 1, f);
  }

  void calc(const float din1[nIn], const float din2[nIn], float dout[nOut]){
    float score_w[nIn][nOut], score[nOut];
    memset(score_w, 0, sizeof(score_w));
    for(int k = 0; k < nOut; k ++){
      score[k] = bias[k];
    }
    for(int i = 0; i < nIn; i ++){
      for(int j = 0; j < nIn; j ++){
        for(int k = 0; k < nOut; k ++){
          score_w[j][k] += din1[i] * W[i][j][k];
        }
      }
    }
    for(int j = 0; j < nIn; j ++){
      for(int k = 0; k < nOut; k ++){
        score[k] += score_w[j][k] * din2[j];
      }
    }

    for(int k = 0; k < nOut; k ++){
      for(int i = 0; i < nIn; i ++){
        score[k] += WB[k][i] * din1[i];
      }
      for(int i = 0; i < nIn; i ++){
        score[k] += WB[k][nIn + i] * din2[i];
      }
    }

    for(int k = 0; k < nOut; k ++){
      dout[k] = fmax(0.f, score[k]);
    }
  }
};

template<int nIn, int nOut>
struct LinearNN{
  float W[nOut][nIn];
  float bias[nOut];

  void load(FILE* f){
    fread(&W, sizeof(W), 1, f);
    fread(&bias, sizeof(bias), 1, f);
  }

  void calc(const float din[nIn], float dout[nOut]){
    for(int k = 0; k < nOut; k ++){
      dout[k] = bias[k];
    }
    for(int i = 0; i < nIn; i ++){
      for(int k = 0; k < nOut; k ++){
        dout[k] += W[k][i] * din[i];
      }
    }
  }
};

struct SingleGraph{
  GCNConv<nFeatures, nFilters1> gcnl1;
  GCNConv<nFilters1, nFilters2> gcnl2;
  GCNConv<nFilters2, nFilters3> gcnl3;
  Average<nFilters3> average;

  void load(FILE* f){
    gcnl1.load(f);
    gcnl2.load(f);
    gcnl3.load(f);
  }

  void process_single(int N, int M, const int G[][2], const int feature[], float pool_feat[nFilters3]){
    float FM[N][nFeatures];
    float L1[N][nFilters1];
    float L2[N][nFilters2];
    float abs_feat[N][nFilters3];
    memset(FM, 0, sizeof(FM));
    for(int i = 0; i < N; i ++){
      FM[i][feature[i]] = 1.0f;
    }
    gcnl1.calc(N, M, G, FM, L1);
    relu(N, L1);
    gcnl2.calc(N, M, G, L1, L2);
    relu(N, L2);
    gcnl3.calc(N, M, G, L2, abs_feat);
    average.calc(N, abs_feat, pool_feat);
    relu(nFilters3, pool_feat);
  }
};

// This is only used to calculate the size of weight
struct CrossComputing {
  TensorNet<nFilters3, nTensorNeurons> tensorNet;
  LinearNN<nScores, nDense1> dense1;
  LinearNN<nDense1, nDense2> dense2;
  LinearNN<nDense2, 1> dense3;

  void load(FILE *f){
    tensorNet.load(f);
    dense1.load(f);
    dense2.load(f);
    dense3.load(f);
  }
};

const int nWeights = sizeof(CrossComputing) / sizeof(float);

#endif
