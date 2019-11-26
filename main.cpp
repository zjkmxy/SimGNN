#include <iostream>
#include <cstdio>
#include <algorithm>
#include <set>
#include <vector>
#include <list>
#include <cmath>
using namespace std;

inline float sigmoid(float x){
  return 1.f / (1.f + exp(- x));
}

template<int B>
inline void relu(int A, float M[][B]){
  for(int i = 0; i < A; i ++){
    for(int j = 0; j < B; j ++){
      M[i][j] = max(0, M[i][j]);
    }
  }
}

template<int nIn, int nOut>
struct GCNConv{
  float W[nIn][nOut];
  float bias[nOut];
  
  GCNConv(float Weight[nIn][nOut], float Bias[nOut]){
    for(int i = 0; i < nIn; i ++){
      for(int j = 0; j < nOut; j ++){
        W[i][j] = Weight[i][j];
      }
    }
    for(int j = 0; j < nOut; j ++){
      bias[j] = Bias[j];
    }
  }
  
  // N = #nodes, M = #edges (including self), G[M][2] (directed), din[N][nIn], dOut[N][nOut]
  void calc(int N, int M, int G[][2], float din[][nIn], float dout[][nOut]){
    int deg[N], degsqrt[N], norm[M];
    int AX[N][nIn];
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
        AX[ed][f] += norm[st] * din[st][f];
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
  Attention(float Weight[F][F]){
    for(int i = 0; i < F; i ++){
      for(int j = 0; j < F; j ++){
        W[i][j] = Weight[i][j];
      }
    }
  }
  
  void calc(int N, float din[][F], float dout[F]){
    float global_context[F], sig_scores[N];
    memset(global_context, 0, sizeof(global_context));
    memset(sig_scores, 0, sizeof(sig_scores));
    for(int i = 0; i < N; i ++){
      for(int f = 0; f < F; f ++){
        global_context[f] += din[i][f];
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

template<int nIn, int nOut>
struct TensorNet{
  float bias[nOut];
  float W[nIn][nIn][nOut];
  float WB[nOut][nIn*2];
  
  TensorNet(float Bias[nOut], float Weight[nIn][nIn][nOut], float WeightBlock[nOut][nIn*2]){
    for(int i = 0; i < nIn; i ++){
      for(int j = 0; j < nIn; j ++){
        for(int k = 0; k < nOut; k ++){
          W[i][j][k] = Weight[i][j][k];
        }
      }
    }
    for(int k = 0; k < nOut; k ++){
      for(int i = 0; i < nIn * 2; i ++){
        WB[k][i] = WeightBlock[k][i];
      }
    }
    for(int k = 0; k < nOut; k ++){
      bias[k] = Bias[k];
    }
  }
  
  void calc(float din1[nIn], float din2[nIn], float dout[nOut]){
    float score_w[nIn][nOut], score[nOut];
    memset(score_w, 0, sizeof(score_w));
    memset(score, 0, sizeof(score));
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
      dout[k] = max(0, score[k] + bias[k]);
    }
  }
};

template<int F, int Bin>
struct Histogram{
  void calc(int N1, int N2, float din1[][F], float din2[][F], float dout[Bin]){
    float scores[N1][N2];
    float minv = 1e30, maxv = 0, dist;
    for(int i = 0; i < N1; i ++){
      for(int j = 0; j < N2; j ++){
        scores[i][j] = 0;
        for(int k = 0; k < F; k ++){
          scores[i][j] += din1[i][k] * din2[j][k];
        }
        minv = min(minv, scores[i][j]);
        maxv = max(maxv, scores[i][j]);
      }
    }
    for(int k = 0; k < Bin; k ++){
      dout[k] = 0;
    }
    dist = (maxv - minv) / Bin;
    for(int i = 0; i < N1; i ++){
      for(int j = 0; j < N2; j ++){
        int pos = floor((scores[i][j] - minv) / dist);
        pos = max(min(pos, Bin - 1), 0);
        dout[pos] += 1;
      }
    }
  }
};

template<int nIn, int nOut>
struct LinearNN{
  float W[nIn][nOut];
  float bias[nOut];
  
  LinearNN(float Bias[nOut], float Weight[nIn][nOut]){
    for(int i = 0; i < nIn; i ++){
      for(int k = 0; k < nOut; k ++){
        W[i][k] = Weight[i][k];
      }
    }
    for(int k = 0; k < nOut; k ++){
      bias[k] = Bias[k];
    }
  }
  
  void calc(float din[nIn], float dout[nOut]){
    for(int k = 0; k < nOut; k ++){
      dout[k] = bias[k];
    }
    for(int i = 0; i < nIn; i ++){
      for(int k = 0; k < nOut; k ++){
        dout[k] += W[i][k] * din[i];
      }
    }
  }
};

static const int nFeatures = 16;
static const int nFilters1 = 128;
static const int nFilters2 = 64;
static const int nFilters3 = 32;
static const int nBin = 16;
static const int nTensorNeurons = 16;
static const int nScores = nBin + nTensorNeurons;
static const int nBottleNeck = 16;

struct SimGNN{
  GCNConv<nFeatures, nFilters1> gcnl1;
  GCNConv<nFilters1, nFilters2> gcnl2;
  GCNConv<nFilters2, nFilters3> gcnl3;
  Attention<nFilters3> attention;
  Histogram<nFilters3, nBin> hist;
  TensorNet<nFilters3, nTensorNeurons> tensorNet;
  LinearNN<nScores, nBottleNeck> firstNN;
  LinearNN<nBottleNeck, 1> scoreNN;
  
  void process_single(int N, int M, int G[][2], int feature[],
                      float abs_feat[][nFilters3], float pool_feat[nFilters3])
  {
    float FM[N][nFeatures];
    float L1[N][nFilters1];
    float L2[N][nFilters2];
    memset(FM, 0, sizeof(FM));
    for(int i = 0; i < N; i ++){
      FM[i][feature[i]] = 1.0f;
    }
    gcnl1.calc(N, M, G, FM, L1);
    relu(N, L1);
    gcnl2.calc(N, M, G, L1, L2);
    relu(N, L2);
    gcnl3.calc(N, M, G, L2, abs_feat);
    attention.calc(N, abs_feat, pool_feat);
  }
  
  float calc_score(int N1, int N2, float abs_feat1[][nFilters3], float abs_feat2[][nFilters3],
                   float pool_feat1[nFilters3], float pool_feat2[nFilters3])
  {
    float scores[nScores], mid[nBottleNeck];
    float ret = 0.0f;
    tensorNet.calc(pool_feat1, pool_feat2, &scores[0]);
    hist.calc(N1, N2, abs_feat1, abs_feat2, &scores[nTensorNeurons]);
    firstNN.calc(scores, mid);
    for(int i = 0; i < nBottleNeck; i ++)
      mid[i] = max(0.0f, mid[i]);
    scoreNN.calc(mid, &ret);
    ret = sigmoid(ret);
    return ret;
  }
  
  inline float unnormalize(int N1, int N2, float score){
    float norm = - log(score);
    return score * (N1 + N2) / 2.f;
  }
  
  inline float run_pair(int N1, int M1, int G1[][2], int feature1[],
                        int N2, int M2, int G2[][2], int feature2[])
  {
    float abs_feat1[N1][nFilters3], abs_feat2[N1][nFilters3];
    float pool_feat1[nFilters3], pool_feat2[nFilters3];
    float score;
    process_single(N1, M1, G1, feature1, abs_feat1, pool_feat1);
    process_single(N2, M2, G2, feature2, abs_feat2, pool_feat2);
    score = calc_score(N1, N2, abs_feat1, abs_feat2, pool_feat1, pool_feat2);
    // return unnormalize(N1, N2, score);
    return score;
  }
};

int main() {
  return 0;
}

