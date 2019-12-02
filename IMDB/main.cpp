#include <iostream>
#include <cstdio>
#include <algorithm>
#include <set>
#include <vector>
#include <list>
#include <cmath>
#include <ctime>
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

static const int nFeatures = 1;
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
  
  void process_single(int N, int M, int G[][2], int feature[], float pool_feat[nFilters3])
  {
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
  
  float calc_score(float pool_feat1[nFilters3], float pool_feat2[nFilters3])
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
    float pool_feat1[nFilters3], pool_feat2[nFilters3];
    float score;
    process_single(N1, M1, G1, feature1, pool_feat1);
    process_single(N2, M2, G2, feature2, pool_feat2);
    score = calc_score(pool_feat1, pool_feat2);
    return score;
  }
};

void read_graph(FILE *f, int &N, int &M, vector<int>& feature, vector<int>& G){
  int a, b;
  
  feature.clear();
  G.clear();
  
  fscanf(f, "%d", &N);
  for(int i = 0; i < N; i ++){
    fscanf(f, "%d", &a);
    // feature.push_back(a - 1);
    feature.push_back(0);
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

static const int N_ROW = 1200;
static const int N_COL = 1200;
int id_row[N_ROW], id_col[N_COL];
float row_emb[N_ROW][nFilters3], col_emb[N_ROW][nFilters3];
float all_scores[N_ROW][N_COL];

int main() {
  char filename[30];
  FILE* f = fopen("model.dat", "rb");
  SimGNN* gnn = new SimGNN(f);
  fclose(f);
  
  f = fopen("train.txt", "r");
  for(int i = 0; i < N_ROW; i ++){
    fscanf(f, "%d", id_row + i);
  }
  fclose(f);
  
  f = fopen("test.txt", "r");
  for(int i = 0; i < N_COL; i ++){
    fscanf(f, "%d", id_col + i);
  }
  fclose(f);
  
  int N, M;
  vector<int> feature, G;
  clock_t start, end, mid;
  
  start = clock();
  for(int i = 0; i < N_ROW; i ++){
    sprintf(filename, "./train/%d.txt", id_row[i]);
    f = fopen(filename, "r");
    read_graph(f, N, M, feature, G);
    fclose(f);
    gnn->process_single(N, M, (GRAPH)G.data(), feature.data(), row_emb[i]);
  }
  for(int i = 0; i < N_COL; i ++){
    sprintf(filename, "./test/%d.txt", id_col[i]);
    f = fopen(filename, "r");
    read_graph(f, N, M, feature, G);
    fclose(f);
    gnn->process_single(N, M, (GRAPH)G.data(), feature.data(), col_emb[i]);
  }
  mid = clock();
  for(int i = 0; i < N_ROW; i ++){
    for(int j = 0; j < N_COL; j ++){
      all_scores[i][j] = gnn->calc_score(row_emb[i], col_emb[j]);
    }
  }
  end = clock();
  
  printf("%.6f\n", all_scores[0][0]);
  printf("TIME: %.3f\n", ((float) (end - start)) / CLOCKS_PER_SEC);
  printf("CALC EMBEDDING: %.3f\n", ((float) (mid - start)) / CLOCKS_PER_SEC);
  printf("PAIRWISE SCORE: %.3f\n", ((float) (end - mid)) / CLOCKS_PER_SEC);
  delete gnn;
  return 0;
}
