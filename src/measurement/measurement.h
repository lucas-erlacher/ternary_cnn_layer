#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <chrono>
#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include <iostream>

#define EPS (1e-2)               // epsilon for comparison
#define CYCLES_PER_NS (4.2)      // processor frequency in GHz

enum TimingType {
  NANO,
  MICRO,
  MILLI
};

struct TestCase {
  const int64_t batch_size;
  const int64_t channels;
  const int64_t height;
  const int64_t width;
  const int64_t num_kernels;
  const int64_t kernel_height;  
  const int64_t kernel_width;
  const int64_t padding;
  const int64_t stride;
  const float relu_alpha;
  const float q_threshold;

  const TimingType timing_type;
};

struct Timing {
  int64_t quantize_ns;
  int64_t img_2_row_ns;
  int64_t gemm_ns;
  int64_t prelu_ns;
};

using TimePoint = std::chrono::high_resolution_clock::time_point;

typedef Timing (*comp_func)(TestCase tc, float* tensor_x, float* tensor_w, float* res);

void add_function(comp_func f, std::string name, size_t ops);
void addPair(comp_func f1, comp_func f2, std::string name);
void register_functions();
void register_function_pairs();

void fill_rands(float* m, int64_t size);
int compare(const void *a, const void *b);
int64_t computeMedian(std::vector<int64_t>& values);
Timing computeMedianTiming(const std::vector<Timing>& timings);
void printArray(int64_t* arr, size_t size);

std::vector<float> convert_NHWC_to_NCHW(float* x, int64_t N, int64_t H, int64_t W, int64_t C);
std::vector<float> convert_NCHW_to_NHWC(float* x, int64_t N, int64_t C, int64_t H, int64_t W);

/* Set matrix to zero */
template<typename T>
void zeros(T* m, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    m[i] = 0;
  }
}

/* Deallocate matrix */
template<typename T>
void destroy(T* m) {
  free(m);
}

/* compares two tensors X1, X2 using NHWC format */
template <typename T>
int compare_tensors(T* X1, T* X2, int64_t N, int64_t C, int64_t H, int64_t W) {
  for (int64_t n = 0; n < N; n++) {
    for (int64_t c = 0; c < C; c++) {
      for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
          // Use N_C_H_W format
          T xx = X1[((n * H + h) * W + w) * C + c] - X2[((n * H + h) * W + w) * C + c];
          if ((xx > EPS) || (xx < -EPS)) {
            return -1;
          }
        }
      }
    }
  }
  return 1;
}

#endif
