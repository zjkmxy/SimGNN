#include <iostream>
#include <cstdio>
#include <algorithm>
#include <set>
#include <vector>
#include <list>
#include <cmath>
using namespace std;

typedef int (*GRAPH)[2];

inline float sigmoid(float x){
  return 1.f / (1.f + exp(- x));
}

template<int B>
inline void relu(int A, float M[][B]){
  for(int i = 0; i < A; i ++){
    for(int j = 0; j < B; j ++){
      M[i][j] = max(0.f, M[i][j]);
    }
  }
}

inline void relu(int N, float F[]){
  for(int i = 0; i < N; i ++){
    F[i] = max(0.f, F[i]);
  }
}

template<int nIn, int nOut>
struct GCNConv{
  float W[nIn][nOut];
  float bias[nOut];
  
  GCNConv(FILE *f){
    fread(&W, sizeof(W), 1, f);
    fread(&bias, sizeof(bias), 1, f);
  }
  
  // N = #nodes, M = #edges (including self), G[M][2] (directed), din[N][nIn], dOut[N][nOut]
  void calc(int N, int M, int G[][2], float din[][nIn], float dout[][nOut]){
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
  Attention(FILE* f){
    fread(&W, sizeof(W), 1, f);
  }
  
  void calc(int N, float din[][F], float dout[F]){
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
  void calc(int N, float din[][F], float dout[F]){
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

template<int nIn, int nOut>
struct TensorNet{
  float bias[nOut];
  float W[nIn][nIn][nOut];
  float WB[nOut][nIn*2];
  
  TensorNet(FILE* f){
    fread(&W, sizeof(W), 1, f);
    fread(&WB, sizeof(WB), 1, f);
    fread(&bias, sizeof(bias), 1, f);
  }
  
  void calc(float din1[nIn], float din2[nIn], float dout[nOut]){
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
      dout[k] = max(0.f, score[k]);
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
    for(int k = 0; k < Bin; k ++){
      dout[k] /= 1.0f * N1 * N2;
    }
  }
};

template<int nIn, int nOut>
struct LinearNN{
  float W[nOut][nIn];
  float bias[nOut];
  
  LinearNN(FILE* f){
    fread(&W, sizeof(W), 1, f);
    fread(&bias, sizeof(bias), 1, f);
  }
  
  void calc(float din[nIn], float dout[nOut]){
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

static const int nFeatures = 29;
static const int nFilters1 = 64;
static const int nFilters2 = 32;
static const int nFilters3 = 16;
// static const int nBin = 16;
static const int nTensorNeurons = 16;
static const int nScores = nTensorNeurons;
static const int nDense1 = 8;
static const int nDense2 = 4;

struct SimGNN{
  GCNConv<nFeatures, nFilters1> gcnl1;
  GCNConv<nFilters1, nFilters2> gcnl2;
  GCNConv<nFilters2, nFilters3> gcnl3;
  Average<nFilters3> average;
  TensorNet<nFilters3, nTensorNeurons> tensorNet;
  LinearNN<nScores, nDense1> dense1;
  LinearNN<nDense1, nDense2> dense2;
  LinearNN<nDense2, 1> dense3;
  
  SimGNN(FILE* f):
    gcnl1(f),
    gcnl2(f),
    gcnl3(f),
    tensorNet(f),
    dense1(f),
    dense2(f),
    dense3(f)
  {}
  
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
    average.calc(N, abs_feat, pool_feat);
    relu(nFilters3, pool_feat);
  }
  
  float calc_score(int N1, int N2, float abs_feat1[][nFilters3], float abs_feat2[][nFilters3],
                   float pool_feat1[nFilters3], float pool_feat2[nFilters3])
  {
    float scores[nScores], l1[nDense1], l2[nDense2];
    float ret = 0.0f;
    tensorNet.calc(pool_feat1, pool_feat2, &scores[0]);
    dense1.calc(scores, l1);
    relu(nDense1, l1);
    dense2.calc(l1, l2);
    relu(nDense2, l2);
    dense3.calc(l2, &ret);
    ret = sigmoid(ret);
    return ret;
  }
  
  inline float unnormalize(int N1, int N2, float score){
    float norm = - log(score);
    return norm * (N1 + N2) / 2.f;
  }
  
  inline float run_pair(int N1, int M1, int G1[][2], int feature1[],
                        int N2, int M2, int G2[][2], int feature2[])
  {
    float abs_feat1[N1][nFilters3], abs_feat2[N2][nFilters3];
    float pool_feat1[nFilters3], pool_feat2[nFilters3];
    float score;
    process_single(N1, M1, G1, feature1, abs_feat1, pool_feat1);
    process_single(N2, M2, G2, feature2, abs_feat2, pool_feat2);
    score = calc_score(N1, N2, abs_feat1, abs_feat2, pool_feat1, pool_feat2);
    // return unnormalize(N1, N2, score);
    return score;
  }
};

void read_graph(FILE *f, int &N, int &M, vector<int>& feature, vector<int>& G){
  int a, b;
  fscanf(f, "%d", &N);
  for(int i = 0; i < N; i ++){
    fscanf(f, "%d", &a);
    feature.push_back(a - 1);
  }
  fscanf(f, "%d", &M);
  for(int i = 0; i < M; i ++){
    fscanf(f, "%d%d", &a, &b);
    G.push_back(a);
    G.push_back(b);
    G.push_back(b);
    G.push_back(a);
  }
  M = M + M + N;
  for(int i = 0; i < N; i ++){
    G.push_back(i);
    G.push_back(i);
  }
}

int main() {
  FILE* f = fopen("model.dat", "rb");
  SimGNN* gnn = new SimGNN(f);
  fclose(f);
  
  int N1, N2, M1, M2;
  vector<int> feature1, G1, feature2, G2;
  f = fopen("graph.txt", "r");
  read_graph(f, N1, M1, feature1, G1);
  read_graph(f, N2, M2, feature2, G2);
  
  float ret = gnn->run_pair(N1, M1, (GRAPH)G1.data(), feature1.data(),
                            N2, M2, (GRAPH)G2.data(), feature2.data());
  printf("%.6f\n", ret);
  
  delete gnn;
  return 0;
}
