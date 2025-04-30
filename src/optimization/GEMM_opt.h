#include "../TAB/common.h"

void TNNGEMM_baseline(int64_t* a, int64_t* b, int* y, size_t M, size_t N, size_t K);
void TNNGEMM_baseline_blocked(int64_t* a, int64_t* b, int* y, size_t M, size_t N, size_t K);
void TNNGEMM_baseline_SIMD(int64_t* a, int64_t* b, float* y, size_t M, size_t N, size_t K, float alpha);
void TNNGEMM_popcount_SIMD_prelu(int64_t* a, int64_t* b, float* y, size_t M, size_t N, size_t K, float alpha);
void TNNGEMM_popcount_SIMD(int64_t* a, int64_t* b, int* y, size_t M, size_t N, size_t K);
void TNNGEMM_baseline_SIMD_unrolled(int64_t* a, int64_t* b, float* y, size_t M, size_t N, size_t K, float alpha);
void TNNGEMM_opt(int64_t* a, int64_t* b, int* y, size_t M, size_t N, size_t K);
