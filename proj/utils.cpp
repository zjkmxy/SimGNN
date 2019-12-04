#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include "ml-layers.hpp"
#include "const.h"

static char filename[255];

void read_graph(FILE *f, int &N, int &M, std::vector<int>& feature, std::vector<int>& G){
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

FILE* open_model(const char* dir){
  sprintf(filename, "%s/model.dat", dir);
  return fopen(filename, "r");
}

void read_ids(const char* dir, const char* name, int N, int ids[]){
  sprintf(filename, "%s/%s.txt", dir, name);
  FILE* f = fopen(filename, "r");
  for(int i = 0; i < N; i ++){
    fscanf(f, "%d", ids + i);
  }
  fclose(f);
}

SingleGraph* create_single(FILE* file){
  SingleGraph* ret = new SingleGraph();
  ret->load(file);
  return ret;
}

void del_single(SingleGraph* p){
  delete p;
}

void calc_embs(SingleGraph* calc, const char* dir, const char* name, float embs[][SIZE_EMBEDDING],
               const int ids[], int size)
{
  FILE *f;
  int N, M;
  std::vector<int> feature, G;

  for(int i = 0; i < size; i ++){
    sprintf(filename, "%s/%s/%d.txt", dir, name, ids[i]);
    f = fopen(filename, "r");
    read_graph(f, N, M, feature, G);
    fclose(f);

    calc->process_single(N, M, (GRAPH)G.data(), feature.data(), embs[i]);
  }
}

void load_weights(FILE *f, float weights[]){
  CrossComputing* p = reinterpret_cast<CrossComputing*>(weights);
  p->load(f);
}
