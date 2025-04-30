#include <list>
#include <vector>
#include <string.h>
#include <iostream>
#include <random>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../optimization/GEMM_opt.h"
#include "../TAB/common.h"

#define CYCLES_PER_NS (4.2)      // processor frequency in GHz

void fill_matrix(int64_t* m, size_t size) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int64_t> distribution;
  for (int i = 0; i < size; i++) {
    m[i] = distribution(generator);
  }
}

// Comparison function for sorting
int compare(const void *a, const void *b) {
  double *da = (double *)a;
  double *db = (double *)b;
  if (*da < *db) return -1;
  if (*da > *db) return 1;
  return 0;
}

// Function to find the median in an array
double findMedian(double arr[], int size) {
  // Sort the array
  qsort(arr, size, sizeof(double), compare);

  // If the size of the array is odd, return the middle element
  if (size % 2 != 0) {
    return arr[size / 2];
  }
  
  // If the size of the array is even, return the average of the two middle elements
  else {
    return (arr[size / 2 - 1] + arr[size / 2]) / 2.0;
  }
}

/*
  Reports and returns the number of cycles required per iteration
*/
double perf_test(const size_t M, const size_t N, const size_t K, int rep, int num, 
                 void (*func)(int64_t*, int64_t*, int*, size_t, size_t, size_t)) {
  double cyclesArray[rep];
  double cycles;

  // Performance measurements repeated rep times.
  // We simply store all results and get median.
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  double duration_ns;
  for (size_t j = 0; j < rep; j++) {
    int64_t* a = (int64_t*) malloc(M * K * BITS * sizeof(int64_t));
    int64_t* b = (int64_t*) malloc(N * K * BITS * sizeof(int64_t));
    int* y = (int*) malloc(M * N * sizeof(int));

    fill_matrix(a, M * K * BITS);
    fill_matrix(b, N * K * BITS);

    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num; i++) {
      func(a, b, y, M, N, K);
    }
    end_time = std::chrono::high_resolution_clock::now();
    cycles = (double) (end_time - start_time).count() * CYCLES_PER_NS;

    cyclesArray[j] = cycles / num;

    free(a);
    free(b);
    free(y);
  }

  return findMedian(cyclesArray, rep);
}

/*
  Reports and returns the number of cycles required per iteration
*/
double perf_test_fused(const size_t M, const size_t N, const size_t K, int rep, int num, 
                 void (*func)(int64_t*, int64_t*, float*, size_t, size_t, size_t, float)) {
  double cyclesArray[rep];
  double cycles;

  // Performance measurements repeated rep times.
  // We simply store all results and get median.
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  double duration_ns;
  for (size_t j = 0; j < rep; j++) {
    int64_t* a = (int64_t*) malloc(M * K * BITS * sizeof(int64_t));
    int64_t* b = (int64_t*) malloc(N * K * BITS * sizeof(int64_t));
    float* y = (float*) malloc(M * N * sizeof(float));

    fill_matrix(a, M * K * BITS);
    fill_matrix(b, N * K * BITS);

    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num; i++) {
      func(a, b, y, M, N, K, 1.0);
    }
    end_time = std::chrono::high_resolution_clock::now();
    cycles = (double) (end_time - start_time).count() * CYCLES_PER_NS;

    cyclesArray[j] = cycles / num;

    free(a);
    free(b);
    free(y);
  }

  return findMedian(cyclesArray, rep);
}

/*
  Performs one performance measurement for a particular GEMM implementation.
  Takes as number of experiments (rep) and number of executions per experiment (num) as parameter.
  Look at the configuration section to specify other parameters.
*/
int main(int argc, char **argv) {
  std::cout << "Starting program. " << std::endl;

  int rep = 1;
  int num = 1;

  if (argc != 3) {
    std::cout << "Running with default arguments. rep = " << rep << ", num = " << num << std::endl;
  } else {
    try {
      // Convert command-line arguments to integers
      rep = std::stoi(argv[1]);
      num = std::stoi(argv[2]);

      if (rep <= 0 || num <= 0) {
        throw std::invalid_argument("Arguments must be positive integers.");
      }

      std::cout << "Running the experiment " << rep << " time(s)." << std::endl;
      std::cout << "Each experiments computes GEMM " << num << " time(s)." << std::endl;
    } catch (const std::invalid_argument& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
    } catch (const std::out_of_range& e) {
      std::cerr << "Error: One of the arguments is out of range." << std::endl;
      return 1;
    }
  }

  /*
    OUTH = (H + 2 * p - KH) / s + 1
    OUTH = (W + 2 * p - KW) / s + 1
    C_P = ⌈C / 64⌉
  */

  /* Configuration Section Start */
  const size_t M = 10000;               // BN * OUTH * OUTW
  const size_t N = 50;                  // KN
  const size_t K = 10000;               // C_P * KH * KW

  // double cycles = perf_test(M, N, K, rep, num, &TNNGEMM_baseline_SIMD);
  double cycles = perf_test_fused(M, N, K, rep, num, &TNNGEMM_baseline_SIMD_unrolled);
  /*  Configuration Section End  */

  const double num_popcnt = (double) M * N * K * 2;

  double pops_per_cycle = num_popcnt / cycles;

  std::cout << cycles << " cycles" << std::endl;
  std::cout << num_popcnt << " popcounts" << std::endl;
  std::cout << pops_per_cycle << " popcounts per cycle" << std::endl;

  return 0;
}