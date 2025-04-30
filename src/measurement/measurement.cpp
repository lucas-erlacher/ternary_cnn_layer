#include "measurement.h"

/* Comparison function for sorting */
int compare(const void *a, const void *b) {
  double *da = (double *)a;
  double *db = (double *)b;
  if (*da < *db) return -1;
  if (*da > *db) return 1;
  return 0;
}

/* Compute the median of a vector of integers */
int64_t computeMedian(std::vector<int64_t>& values) {
  size_t size = values.size();
  if (size == 0) {
    throw std::domain_error("Median of an empty array is undefined.");
  }

  std::sort(values.begin(), values.end());

  if (size % 2 == 0) {
    return (values[size / 2 - 1] + values[size / 2]) / 2;
  }
  return values[size / 2];
}

/* Compute the median of timings */
Timing computeMedianTiming(const std::vector<Timing>& timings) {
  if (timings.empty()) {
    throw std::domain_error("Median of an empty array is undefined.");
  }

  std::vector<int64_t> quantize_ns;
  std::vector<int64_t> img_2_row_ns;
  std::vector<int64_t> gemm_ns;
  std::vector<int64_t> prelu_ns;

  for (const Timing& timing : timings) {
    quantize_ns.push_back(timing.quantize_ns);
    img_2_row_ns.push_back(timing.img_2_row_ns);
    gemm_ns.push_back(timing.gemm_ns);
    prelu_ns.push_back(timing.prelu_ns);
  }

  int64_t median_quantize_ns = computeMedian(quantize_ns);
  int64_t median_img_2_row_ns = computeMedian(img_2_row_ns);
  int64_t median_gemm_ns = computeMedian(gemm_ns);
  int64_t median_prelu_ns = computeMedian(prelu_ns);

  return {median_quantize_ns, median_img_2_row_ns, median_gemm_ns, median_prelu_ns};
}

/* Create random inputs */
void fill_rands(float* m, int64_t size) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(-1, 1);

  for (size_t i = 0; i < size; i++) {
    m[i] = distribution(generator);
  }
}

/* Reshaping functions */
std::vector<float> convert_NHWC_to_NCHW(float* x, int64_t N, int64_t H, int64_t W, int64_t C) {
  std::vector<float> cvt_x = std::vector<float>(N * C * H * W);
  for (int64_t n = 0; n < N; n++) {
    for(int64_t h = 0; h < H; h++) {
      for(int64_t w = 0; w < W; w++) {
        for(int64_t c = 0; c < C; c++) {
          cvt_x[n * (C * H * W) + c * (H * W) + h * W + w] = x[n * (H * W * C) + h * (W * C) + w * C + c];
        }
      }
    }
  }
  return cvt_x;
}

std::vector<float> convert_NCHW_to_NHWC(float* x, int64_t N, int64_t C, int64_t H, int64_t W) {
  std::vector<float> cvt_x = std::vector<float>(N * H * W * C);
  
  for(int64_t n = 0; n < N; n++) {
    for(int64_t c = 0; c < C; c++) {
      for(int64_t h = 0; h < H; h++) {
        for(int64_t w = 0; w < W; w++) {
          cvt_x[n * (H * W * C) + h * (W * C) + w * C + c] = x[n * (C * H * W) + c * (H * W) + h * W + w];
        }
      }
    }
  }

  return cvt_x;
}

/* pritty print an array of measurements */
void printArray(int64_t* arr, size_t size) {
  std::cout << "[";
  if (size > 0) {
    std::cout << arr[0];
    for (size_t i = 1; i < size; ++i) {
      std::cout << ", " << arr[i];
    }
  }
  std::cout << "]" << std::endl;
}