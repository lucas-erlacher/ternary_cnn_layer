#include "GEMM_opt.h"

#define BLOCK_SIZE 32

// a = M * K * 2, b = N * K * 2, y = M * N
// Baseline as in the original code
void TNNGEMM_baseline(int64_t* a, int64_t* b, int* y, size_t M, size_t N, size_t K) {
  const int KB = K * BITS;

  for (size_t oh = 0; oh < M; oh++) {
    for (size_t ow = 0; ow < N; ow++) {
      int cntp1 = 0;
      int cntp2 = 0;
      for (size_t ik = 0; ik < KB; ik += BITS) {
        // Use H_W_B format
        int64_t p1 = a[oh * KB + ik + 0] ^ b[ow * KB + ik + 0];
        int64_t p2 = a[oh * KB + ik + 1] & b[ow * KB + ik + 1];
        int64_t p3 = p1 & p2;
        cntp1 = cntp1 + popcnt64(p3);
        cntp2 = cntp2 + popcnt64(p2);
      }
      y[oh * N + ow] = cntp2 - cntp1 - cntp1;
    }
  }
}

// Blocked version of baseline using a 32x32 block size to perform the dot product
void TNNGEMM_baseline_blocked(int64_t* a, int64_t* b, int* y, size_t M, size_t N, size_t K) {
  const size_t KB = K * BITS;

  // std::cout << "M:" << M << ", N:" << N << ", K:" << K << std::endl;
  // M:100352, N:64, K:9

  for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
    for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
      for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, M); i++) {
        for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); j++) {
          size_t k = 0;
          int cntp11 = 0;
          int cntp12 = 0;

          int cntp21 = 0;
          int cntp22 = 0;

          int cntp31 = 0;
          int cntp32 = 0;
          for (k = 0; k < KB; k += 3 * BITS) {
            int64_t p11 = a[i * KB + k + 0] ^ b[j * KB + k + 0];
            int64_t p12 = a[i * KB + k + 1] & b[j * KB + k + 1];
            int64_t p13 = p11 & p12;
            cntp11 = cntp11 + popcnt64(p13);
            cntp12 = cntp12 + popcnt64(p12);

            int64_t p21 = a[i * KB + k + 2] ^ b[j * KB + k + 2];
            int64_t p22 = a[i * KB + k + 3] & b[j * KB + k + 3];
            int64_t p23 = p21 & p22;
            cntp21 = cntp21 + popcnt64(p23);
            cntp22 = cntp22 + popcnt64(p22);

            int64_t p31 = a[i * KB + k + 4] ^ b[j * KB + k + 4];
            int64_t p32 = a[i * KB + k + 5] & b[j * KB + k + 5];
            int64_t p33 = p31 & p32;
            cntp31 = cntp31 + popcnt64(p33);
            cntp32 = cntp32 + popcnt64(p32);
          }

          for (; k < KB; k += 2 * BITS) {
            int64_t p11 = a[i * KB + k] ^ b[j * KB + k];
            int64_t p12 = a[i * KB + k + 1] & b[j * KB + k + 1];
            int64_t p13 = p11 & p12;
            cntp11 = cntp11 + popcnt64(p13);
            cntp12 = cntp12 + popcnt64(p12);
          }

          y[i * N + j] = y[i * N + j] + cntp12 - cntp11 - cntp11
                                      + cntp22 - cntp21 - cntp21
                                      + cntp32 - cntp31 - cntp31;
        }
      }
    }
  }
}

// Unrolling the inner loop and vecotrizing the applicable computations using simd
void TNNGEMM_baseline_SIMD(int64_t* a, int64_t* b, float* y, size_t M, size_t N, size_t K, float alpha) {
  const size_t KB = K * BITS;

  // std::cout << "M:" << M << ", N:" << N << ", K:" << K << std::endl;
  // M:100352, N:64, K:9
  int64_t* p2_vals = static_cast<int64_t*>(aligned_alloc(32, 4 * sizeof(int64_t)));
  int64_t* p3_vals = static_cast<int64_t*>(aligned_alloc(32, 4 * sizeof(int64_t)));
  
  for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
    for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
      for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, M); i++) {
        for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); j++) {
          size_t k = 0;
          int cntp11 = 0;
          int cntp12 = 0;
          

          for (k = 0; k < KB - 7; k += 4 * BITS) {
            //Load section
            __m256i va1 = _mm256_loadu_si256((__m256i*) (a + i * KB + k));
            __m256i vb1 = _mm256_loadu_si256((__m256i*) (b + j * KB + k));

            __m256i va2 = _mm256_loadu_si256((__m256i*) (a + i * KB + k + 4));
            __m256i vb2 = _mm256_loadu_si256((__m256i*) (b + j * KB + k + 4));

            __m256i va3 = _mm256_unpacklo_epi64(va1, va2);
            __m256i vb3 = _mm256_unpacklo_epi64(vb1, vb2);

            __m256i va4 = _mm256_unpackhi_epi64(va1, va2);
            __m256i vb4 = _mm256_unpackhi_epi64(vb1, vb2);

            // Compute section
            __m256i p1 = _mm256_xor_si256(va3, vb3);
            __m256i p2 = _mm256_and_si256(va4, vb4);
            __m256i p3 = _mm256_and_si256(p1, p2);

            // Store section
            _mm256_store_si256((__m256i*) p2_vals, p2);
            _mm256_store_si256((__m256i*) p3_vals, p3);

            // non vectorized popcounts
            cntp11 = cntp11 + popcnt64(p3_vals[0]);
            cntp11 = cntp11 + popcnt64(p3_vals[1]);
            cntp11 = cntp11 + popcnt64(p3_vals[2]);
            cntp11 = cntp11 + popcnt64(p3_vals[3]);

            cntp12 = cntp12 + popcnt64(p2_vals[0]);
            cntp12 = cntp12 + popcnt64(p2_vals[1]);
            cntp12 = cntp12 + popcnt64(p2_vals[2]);
            cntp12 = cntp12 + popcnt64(p2_vals[3]);
          }

          for (; k < KB; k += 2 * BITS) {
            int64_t p11 = a[i * KB + k] ^ b[j * KB + k];
            int64_t p12 = a[i * KB + k + 1] & b[j * KB + k + 1];
            int64_t p13 = p11 & p12;
            cntp11 = cntp11 + popcnt64(p13);
            cntp12 = cntp12 + popcnt64(p12);
          }

          int temp = cntp12 - cntp11 - cntp11;
          
          // PreLU fusing
          if(temp > 0){
            y[i * N + j] = temp;
          }else{
            y[i * N + j] = temp * alpha;
          }
        }
      }
    }
  }
}

// Unroll factor of 4 of the previous simd version
void TNNGEMM_baseline_SIMD_unrolled(int64_t* a, int64_t* b, float* y, size_t M, size_t N, size_t K, float alpha) {
  const size_t KB = K * BITS;

  // std::cout << "M:" << M << ", N:" << N << ", K:" << K << std::endl;
  // M:100352, N:64, K:9
  int64_t* p2_vals = static_cast<int64_t*>(aligned_alloc(32, 8 * sizeof(int64_t)));
  int64_t* p3_vals = static_cast<int64_t*>(aligned_alloc(32, 8 * sizeof(int64_t)));
  
  for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
    for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
      for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, M); i++) {
        for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); j++) {
          size_t k = 0;
          int cntp11 = 0;
          int cntp12 = 0;

          for (k = 0; k < KB - 15; k += 8 * BITS) {
            __m256i va11 = _mm256_loadu_si256((__m256i*) (a + i * KB + k));
            __m256i vb11 = _mm256_loadu_si256((__m256i*) (b + j * KB + k));

            __m256i va12 = _mm256_loadu_si256((__m256i*) (a + i * KB + k + 4));
            __m256i vb12 = _mm256_loadu_si256((__m256i*) (b + j * KB + k + 4));

            __m256i va13 = _mm256_unpacklo_epi64(va11, va12);
            __m256i vb13 = _mm256_unpacklo_epi64(vb11, vb12);

            __m256i va14 = _mm256_unpackhi_epi64(va11, va12);
            __m256i vb14 = _mm256_unpackhi_epi64(vb11, vb12);

            __m256i p11 = _mm256_xor_si256(va13, vb13);
            __m256i p12 = _mm256_and_si256(va14, vb14);
            __m256i p13 = _mm256_and_si256(p11, p12);

            _mm256_store_si256((__m256i*) p2_vals, p12);
            _mm256_store_si256((__m256i*) p3_vals, p13);

            cntp11 = cntp11 + popcnt64(p3_vals[0]);
            cntp11 = cntp11 + popcnt64(p3_vals[1]);
            cntp11 = cntp11 + popcnt64(p3_vals[2]);
            cntp11 = cntp11 + popcnt64(p3_vals[3]);

            cntp12 = cntp12 + popcnt64(p2_vals[0]);
            cntp12 = cntp12 + popcnt64(p2_vals[1]);
            cntp12 = cntp12 + popcnt64(p2_vals[2]);
            cntp12 = cntp12 + popcnt64(p2_vals[3]);

            __m256i va21 = _mm256_loadu_si256((__m256i*) (a + i * KB + k + 8));
            __m256i vb21 = _mm256_loadu_si256((__m256i*) (b + j * KB + k + 8));

            __m256i va22 = _mm256_loadu_si256((__m256i*) (a + i * KB + k + 12));
            __m256i vb22 = _mm256_loadu_si256((__m256i*) (b + j * KB + k + 12));

            __m256i va23 = _mm256_unpacklo_epi64(va21, va22);
            __m256i vb23 = _mm256_unpacklo_epi64(vb21, vb22);

            __m256i va24 = _mm256_unpackhi_epi64(va21, va22);
            __m256i vb24 = _mm256_unpackhi_epi64(vb21, vb22);

            __m256i p21 = _mm256_xor_si256(va23, vb23);
            __m256i p22 = _mm256_and_si256(va24, vb24);
            __m256i p23 = _mm256_and_si256(p21, p22);

            _mm256_store_si256((__m256i*) (p2_vals + 4), p22);
            _mm256_store_si256((__m256i*) (p3_vals + 4), p23);

            cntp11 = cntp11 + popcnt64(p3_vals[4]);
            cntp11 = cntp11 + popcnt64(p3_vals[5]);
            cntp11 = cntp11 + popcnt64(p3_vals[6]);
            cntp11 = cntp11 + popcnt64(p3_vals[7]);

            cntp12 = cntp12 + popcnt64(p2_vals[4]);
            cntp12 = cntp12 + popcnt64(p2_vals[5]);
            cntp12 = cntp12 + popcnt64(p2_vals[6]);
            cntp12 = cntp12 + popcnt64(p2_vals[7]);
          }

          for (; k < KB; k += 2 * BITS) {
            int64_t p11 = a[i * KB + k] ^ b[j * KB + k];
            int64_t p12 = a[i * KB + k + 1] & b[j * KB + k + 1];
            int64_t p13 = p11 & p12;
            cntp11 = cntp11 + popcnt64(p13);
            cntp12 = cntp12 + popcnt64(p12);
          }

          int temp = cntp12 - cntp11 - cntp11;

          // PreLu fusing 
          if(temp > 0){
            y[i * N + j] = temp;
          }else{
            y[i * N + j] = temp * alpha;
          }
        }
      }
    }
  }
}

// See version below for further comments
void TNNGEMM_popcount_SIMD_prelu(int64_t* a, int64_t* b, float* y, size_t M, size_t N, size_t K,float alpha) {
  const size_t KB = K * BITS;

  // std::cout << "M:" << M << ", N:" << N << ", K:" << K << std::endl;
  // M:100352, N:64, K:9
  const int64_t* p2_vals = static_cast<int64_t*>(aligned_alloc(32, 8 * sizeof(int64_t)));
  const int64_t* p3_vals = static_cast<int64_t*>(aligned_alloc(32, 8 * sizeof(int64_t)));
  std::vector<int64_t> popcount_buffer1 = std::vector<int64_t>(1000);
  std::vector<int64_t> popcount_buffer2 = std::vector<int64_t>(1000);
  
  for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
    for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
      for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, M); i++) {
        for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); j++) {
          size_t k = 0;
          int cntp11 = 0;
          int cntp12 = 0;

          for (k = 0; k < KB - 15; k += 8 * BITS) {
            __m256i va11 = _mm256_loadu_si256((__m256i*) (a + i * KB + k));
            __m256i vb11 = _mm256_loadu_si256((__m256i*) (b + j * KB + k));

            __m256i va12 = _mm256_loadu_si256((__m256i*) (a + i * KB + k + 4));
            __m256i vb12 = _mm256_loadu_si256((__m256i*) (b + j * KB + k + 4));

            __m256i va21 = _mm256_loadu_si256((__m256i*) (a + i * KB + k + 8));
            __m256i vb21 = _mm256_loadu_si256((__m256i*) (b + j * KB + k + 8));

            __m256i va22 = _mm256_loadu_si256((__m256i*) (a + i * KB + k + 12));
            __m256i vb22 = _mm256_loadu_si256((__m256i*) (b + j * KB + k + 12));

            __m256i va13 = _mm256_unpacklo_epi64(va11, va12);
            __m256i vb13 = _mm256_unpacklo_epi64(vb11, vb12);

            __m256i va14 = _mm256_unpackhi_epi64(va11, va12);
            __m256i vb14 = _mm256_unpackhi_epi64(vb11, vb12);

            __m256i va23 = _mm256_unpacklo_epi64(va21, va22);
            __m256i vb23 = _mm256_unpacklo_epi64(vb21, vb22);

            __m256i va24 = _mm256_unpackhi_epi64(va21, va22);
            __m256i vb24 = _mm256_unpackhi_epi64(vb21, vb22);

            __m256i p21 = _mm256_xor_si256(va23, vb23);
            __m256i p22 = _mm256_and_si256(va24, vb24);
            __m256i p23 = _mm256_and_si256(p21, p22);

            __m256i p11 = _mm256_xor_si256(va13, vb13);
            __m256i p12 = _mm256_and_si256(va14, vb14);
            __m256i p13 = _mm256_and_si256(p11, p12);

            _mm256_storeu_si256((__m256i*) &popcount_buffer1[k/4], p12);
            _mm256_storeu_si256((__m256i*) &popcount_buffer2[k/4], p13);

            _mm256_store_si256((__m256i*) (p2_vals), p22);
            _mm256_store_si256((__m256i*) (p3_vals), p23);
            
            cntp11 = cntp11 + popcnt64(p3_vals[0]);
            cntp11 = cntp11 + popcnt64(p3_vals[1]);
            cntp11 = cntp11 + popcnt64(p3_vals[2]);
            cntp11 = cntp11 + popcnt64(p3_vals[3]);
            
            cntp12 = cntp12 + popcnt64(p2_vals[0]);
            cntp12 = cntp12 + popcnt64(p2_vals[1]);
            cntp12 = cntp12 + popcnt64(p2_vals[2]);
            cntp12 = cntp12 + popcnt64(p2_vals[3]);
            
          }

          for (; k < KB; k += 2 * BITS) {
            int64_t p11 = a[i * KB + k] ^ b[j * KB + k];
            int64_t p12 = a[i * KB + k + 1] & b[j * KB + k + 1];
            int64_t p13 = p11 & p12;
            cntp11 = cntp11 + popcnt64(p13);
            cntp12 = cntp12 + popcnt64(p12);
          }

          cntp11 += popcnt(popcount_buffer2.data(), popcount_buffer2.size()*8);
          cntp12 += popcnt(popcount_buffer1.data(), popcount_buffer1.size()*8);

          int temp = cntp12 - cntp11 - cntp11;

          // PreLu fusing 
          if(temp > 0){
            y[i * N + j] = temp;
          }else{
            y[i * N + j] = temp * alpha;
          }
        }
      }
    }
  }
}

// Use the popcount library to try and vectorize popcnt instrcution using AVX2
void TNNGEMM_popcount_SIMD(int64_t* a, int64_t* b, int* y, size_t M, size_t N, size_t K) {
  const size_t KB = K * BITS;

  // std::cout << "M:" << M << ", N:" << N << ", K:" << K << std::endl;
  // M:100352, N:64, K:9
  const int64_t* p2_vals = static_cast<int64_t*>(aligned_alloc(32, 8 * sizeof(int64_t)));
  const int64_t* p3_vals = static_cast<int64_t*>(aligned_alloc(32, 8 * sizeof(int64_t)));
  std::vector<int64_t> popcount_buffer1 = std::vector<int64_t>(K);
  std::vector<int64_t> popcount_buffer2 = std::vector<int64_t>(K);
  
  for (size_t ii = 0; ii < M; ii += BLOCK_SIZE) {
    for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
      for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, M); i++) {
        for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); j++) {
          size_t k = 0;
          int cntp11 = 0;
          int cntp12 = 0;

          // Unroll factor of 4 since we want to be able to use the new popcnt
          // next to the normal popcnt64 to utilize both execution units
          for (k = 0; k < KB - 15; k += 8 * BITS) {

            // Load section
            __m256i va11 = _mm256_loadu_si256((__m256i*) (a + i * KB + k));
            __m256i vb11 = _mm256_loadu_si256((__m256i*) (b + j * KB + k));

            __m256i va12 = _mm256_loadu_si256((__m256i*) (a + i * KB + k + 4));
            __m256i vb12 = _mm256_loadu_si256((__m256i*) (b + j * KB + k + 4));

            __m256i va21 = _mm256_loadu_si256((__m256i*) (a + i * KB + k + 8));
            __m256i vb21 = _mm256_loadu_si256((__m256i*) (b + j * KB + k + 8));

            __m256i va22 = _mm256_loadu_si256((__m256i*) (a + i * KB + k + 12));
            __m256i vb22 = _mm256_loadu_si256((__m256i*) (b + j * KB + k + 12));

            __m256i va13 = _mm256_unpacklo_epi64(va11, va12);
            __m256i vb13 = _mm256_unpacklo_epi64(vb11, vb12);

            __m256i va14 = _mm256_unpackhi_epi64(va11, va12);
            __m256i vb14 = _mm256_unpackhi_epi64(vb11, vb12);

            __m256i va23 = _mm256_unpacklo_epi64(va21, va22);
            __m256i vb23 = _mm256_unpacklo_epi64(vb21, vb22);

            __m256i va24 = _mm256_unpackhi_epi64(va21, va22);
            __m256i vb24 = _mm256_unpackhi_epi64(vb21, vb22);

            // Compute section
            __m256i p21 = _mm256_xor_si256(va23, vb23);
            __m256i p22 = _mm256_and_si256(va24, vb24);
            __m256i p23 = _mm256_and_si256(p21, p22);

            __m256i p11 = _mm256_xor_si256(va13, vb13);
            __m256i p12 = _mm256_and_si256(va14, vb14);
            __m256i p13 = _mm256_and_si256(p11, p12);

            // Store section
            _mm256_storeu_si256((__m256i*) &popcount_buffer1[k/4], p12);
            _mm256_storeu_si256((__m256i*) &popcount_buffer2[k/4], p13);

            _mm256_store_si256((__m256i*) (p2_vals), p22);
            _mm256_store_si256((__m256i*) (p3_vals), p23);

            // non vectorized popcnt calculation
            cntp11 = cntp11 + popcnt64(p3_vals[0]);
            cntp11 = cntp11 + popcnt64(p3_vals[1]);
            cntp11 = cntp11 + popcnt64(p3_vals[2]);
            cntp11 = cntp11 + popcnt64(p3_vals[3]);
            
            cntp12 = cntp12 + popcnt64(p2_vals[0]);
            cntp12 = cntp12 + popcnt64(p2_vals[1]);
            cntp12 = cntp12 + popcnt64(p2_vals[2]);
            cntp12 = cntp12 + popcnt64(p2_vals[3]);
            
          }

          for (; k < KB; k += 2 * BITS) {
            int64_t p11 = a[i * KB + k] ^ b[j * KB + k];
            int64_t p12 = a[i * KB + k + 1] & b[j * KB + k + 1];
            int64_t p13 = p11 & p12;
            cntp11 = cntp11 + popcnt64(p13);
            cntp12 = cntp12 + popcnt64(p12);
          }
          // vectorized popcnt calculation
          cntp11 += popcnt(popcount_buffer2.data(), popcount_buffer2.size()*8);
          cntp12 += popcnt(popcount_buffer1.data(), popcount_buffer1.size()*8);

          int temp = cntp12 - cntp11 - cntp11;

          // PreLu fusing 
          // if(temp > 0){
            y[i * N + j] = temp;
          // }else{
            // y[i * N + j] = temp * alpha;
          // }
        }
      }
    }
  }
}

void TNNGEMM_opt(int64_t* a, int64_t* b, int* y, size_t M, size_t N, size_t K) {
  const size_t KB = K * BITS;
  int cntp1;
  int cntp2;

  if (N > M) {
    for (size_t ow = 0; ow < N; ow++) {
      for (size_t oh = 0; oh < M; oh++) {
        cntp1 = 0;
        cntp2 = 0;
        for (size_t ik = 0; ik < KB; ik += BITS) {
          // Use H_W_B format
          int64_t p1 = a[oh * KB + ik + 0] ^ b[ow * KB + ik + 0];
          int64_t p2 = a[oh * KB + ik + 1] & b[ow * KB + ik + 1];
          int64_t p3 = p1 & p2;
          cntp1 = cntp1 + popcnt64(p3);
          cntp2 = cntp2 + popcnt64(p2);
        }
        y[oh * N + ow] = cntp2 - cntp1 - cntp1;
      }
    }
  } else {
    for (size_t oh = 0; oh < M; oh++) {
      for (size_t ow = 0; ow < N; ow++) {
        cntp1 = 0;
        cntp2 = 0;
        for (size_t ik = 0; ik < KB; ik += BITS) {
          // Use H_W_B format
          int64_t p1 = a[oh * KB + ik + 0] ^ b[ow * KB + ik + 0];
          int64_t p2 = a[oh * KB + ik + 1] & b[ow * KB + ik + 1];
          int64_t p3 = p1 & p2;
          cntp1 = cntp1 + popcnt64(p3);
          cntp2 = cntp2 + popcnt64(p2);
        }
        y[oh * N + ow] = cntp2 - cntp1 - cntp1;
      }
    }
  }
}
