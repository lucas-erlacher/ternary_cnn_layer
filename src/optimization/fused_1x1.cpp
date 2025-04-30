#include "fused_1x1.h"

/*
  SIMD packing.
*/
static void inline pack_simd(float* tensor_in, int64_t* tensor_out, int ik, int channels, int priChannel, int packC, float q_threshold, int64_t* onebit) {
  for (int64_t ic = 0; ic < priChannel; ic++) {
    // for 2-bit packing of channel
    int64_t w1 = 0;
    int64_t w2 = 0;

    const __m256 threshold = _mm256_set1_ps(q_threshold);
    const __m256 neg_threshold = _mm256_set1_ps(-q_threshold);
    for (int64_t bit = 0; bit < cntbits; bit += 8) {
      __m256 currentx = _mm256_loadu_ps(tensor_in + ik * channels + ic * cntbits + bit); // for general C not aligned

      __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
      __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

      int64_t gt_mask = _mm256_movemask_ps(mask_gt);
      int64_t lt_mask = _mm256_movemask_ps(mask_lt);

      w1 |= (lt_mask << bit);
      w2 |= (gt_mask << bit);
      w2 |= (lt_mask << bit);
    }

      // Store the ternarized and packed data in N_H_W_C_B format
      tensor_out[(ik * packC + ic) * BITS + 0] = w1;
      tensor_out[(ik * packC + ic) * BITS + 1] = w2;
    }

    // Pack the second part: priChannel*cntbits ~ C
    if ((channels % cntbits) > 0) {
      int64_t w1 = 0;
      int64_t w2 = 0;

      const __m256 threshold = _mm256_set1_ps(q_threshold);
      const __m256 neg_threshold = _mm256_set1_ps(-q_threshold);

      int64_t bit = 0;
      for (; bit <= (channels % cntbits) - 8; bit += 8) {
        __m256 currentx = _mm256_loadu_ps(tensor_in + ik * channels + priChannel * cntbits + bit);

        __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
        __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

        int64_t gt_mask = _mm256_movemask_ps(mask_gt);
        int64_t lt_mask = _mm256_movemask_ps(mask_lt);

        w1 |= (lt_mask << bit);
        w2 |= (gt_mask << bit);
        w2 |= (lt_mask << bit);
      }
      for (; bit < (channels % cntbits); bit++) {
        float currentx = tensor_in[ik * channels + priChannel * cntbits + bit];
        if (currentx > q_threshold) {
          // Pack 1: 01
          w2 = w2 | onebit[bit];
        } else if (currentx < (-q_threshold)) {
          // Pack -1: 11
          w1 = w1 | onebit[bit];
          w2 = w2 | onebit[bit];
        }
      }

      // Store packed data into new NHWCB format
      tensor_out[(ik * packC + priChannel) * BITS + 0] = w1;
      tensor_out[(ik * packC + priChannel) * BITS + 1] = w2;
    }
}

static void inline pack_buffer(float* tensor_in, int64_t* tensor_out, int in, int channels, int priChannel, int packC, float q_threshold, int64_t* onebit) {
  for (int64_t ic = 0; ic < priChannel; ic++) {
    // for 2-bit packing of channel
    int64_t x1 = 0;
    int64_t x2 = 0;

    const __m256 threshold = _mm256_set1_ps(q_threshold);
    const __m256 neg_threshold = _mm256_set1_ps(-q_threshold);
    for (int64_t bit = 0; bit < cntbits; bit += 8) {
      __m256 currentx = _mm256_loadu_ps(tensor_in + in * channels + ic * cntbits + bit); // for general C not aligned

      __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
      __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

      int64_t gt_mask = _mm256_movemask_ps(mask_gt);
      int64_t lt_mask = _mm256_movemask_ps(mask_lt);

      x1 |= (lt_mask << bit);
      x2 |= (gt_mask << bit);
      x2 |= (lt_mask << bit);
    }

    tensor_out[ic * BITS + 0] = x1;
    tensor_out[ic * BITS + 1] = x2;
  }

  // Pack the second part: priChannel*cntbits ~ C
  if ((channels % cntbits) > 0) {
    int64_t x1 = 0;
    int64_t x2 = 0;

    const __m256 threshold = _mm256_set1_ps(q_threshold);
    const __m256 neg_threshold = _mm256_set1_ps(-q_threshold);

    int64_t bit = 0;
    for (; bit <= (channels % cntbits) - 8; bit += 8) {
      __m256 currentx = _mm256_loadu_ps(tensor_in + in * channels + priChannel * cntbits + bit);

      __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
      __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

      int64_t gt_mask = _mm256_movemask_ps(mask_gt);
      int64_t lt_mask = _mm256_movemask_ps(mask_lt);

      x1 |= (lt_mask << bit);
      x2 |= (gt_mask << bit);
      x2 |= (lt_mask << bit);
    }
    for (; bit < (channels % cntbits); bit++) {
      float currentx = tensor_in[in * channels + priChannel * cntbits + bit];
      if (currentx > q_threshold) {
        // Pack 1: 01
        x2 = x2 | onebit[bit];
      } else if (currentx < (-q_threshold)) {
        // Pack -1: 11
        x1 = x1 | onebit[bit];
        x2 = x2 | onebit[bit];
      }
    }

    tensor_out[priChannel * BITS + 0] = x1;
    tensor_out[priChannel * BITS + 1] = x2;
  }
}

/*
  Specialized 1x1 convolution version 1.
  Fuses all TNN steps into 1 function. Uses a buffer for the data and a buffer for each kernel. Applies SIMD to pack all values.

  Attention: Assumes H = W = kh = kw = 1, padding = 0, stride = 1
*/
void specialized_1x1_v1(float* tensor_x, float* tensor_w, float* res, int batch_size, int channels, int num_kernels, float q_threshold, float relu_alpha) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int priChannel = channels / cntbits;

  // packC: actual packed input channel
  const int packC = (channels % cntbits) ? (priChannel + 1) : priChannel;

  // The quantized weights, in N_H_W_C_B format
  std::vector<int64_t> qx = std::vector<int64_t>(batch_size * packC * BITS, 0);  // maybe don't need 0
  int64_t* qxptr = qx.data();

  for (int64_t ik = 0; ik < batch_size; ik++) {
    pack_simd(tensor_x, qxptr, ik, channels, priChannel, packC, q_threshold, onebit);
  }

  std::vector<int> yi = std::vector<int>(batch_size * num_kernels);

  for (int64_t in = 0; in < num_kernels; in++) {
    std::vector<int64_t> buffer = std::vector<int64_t>(packC * BITS, 0);
    pack_buffer(tensor_w, buffer.data(), in, channels, priChannel, packC, q_threshold, onebit);

    for (int64_t ik = 0; ik < batch_size; ik++) {
      // GEMM
      int cntp1 = 0;
      int cntp2 = 0;

      for (int64_t ic = 0; ic < packC; ic++) {
        int64_t x1 = qxptr[ik * packC * BITS + ic * BITS + 0];
        int64_t x2 = qxptr[ik * packC * BITS + ic * BITS + 1];

        int64_t p1 = buffer[ic * BITS + 0] ^ x1;
        int64_t p2 = buffer[ic * BITS + 1] & x2;

        int64_t p3 = p1 & p2;

        cntp1 = cntp1 + popcnt64(p3);
        cntp2 = cntp2 + popcnt64(p2);
      }

      int temp = cntp2 - cntp1 - cntp1;

      if (temp > 0) {
        res[ik * num_kernels + in] = temp;
      } else {
        res[ik * num_kernels + in] = temp * relu_alpha;
      }
    }
  }
}

/*
  Specialized 1x1 convolution version 2 building on top of version 1.
  Uses AVX2 and popcount in alternating fashion. Works best if batch_size <= num_kernels due to the buffer size.

  Attention: Assumes H = W = kh = kw = 1, padding = 0, stride = 1
*/
void specialized_1x1_v2(float* tensor_x, float* tensor_w, float* res, int batch_size, int channels, int num_kernels, float q_threshold, float relu_alpha) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int64_t priChannel = channels / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (channels % cntbits) ? (priChannel + 1) : priChannel;

  int64_t blocking_size = 256;
  std::vector<int64_t> popcnt_buffer1 = std::vector<int64_t>(blocking_size);
  std::vector<int64_t> popcnt_buffer2 = std::vector<int64_t>(blocking_size);

  std::vector<int64_t> qx = std::vector<int64_t>(batch_size * packC * BITS, 0);  // maybe don't need 0
  int64_t* qxptr = qx.data();

  for (int64_t ik = 0; ik < batch_size; ik++) {
    pack_simd(tensor_x, qxptr, ik, channels, priChannel, packC, q_threshold, onebit);
  }

  std::vector<int> yi = std::vector<int>(batch_size * num_kernels);

  for (int64_t in = 0; in < num_kernels; in++) {
    std::vector<int64_t> buffer = std::vector<int64_t>(packC * BITS, 0);
    pack_buffer(tensor_w, buffer.data(), in, channels, priChannel, packC, q_threshold, onebit);

    for (int64_t ik = 0; ik < batch_size; ik++) {
      // GEMM
      int cntp1 = 0;
      int cntp2 = 0;

      for (int64_t bc = 0; bc < packC; bc += blocking_size * 2) {
        int64_t block_size = std::min(blocking_size * 2, packC - bc);

        int64_t ic = 0;
        for (; ic < block_size - 7; ic += 8) {
          int64_t offset_c = bc + ic;
          __m256i va1 = _mm256_loadu_si256((__m256i*) (&buffer[offset_c * BITS + 0]));
          __m256i vb1 = _mm256_loadu_si256((__m256i*) (&qxptr[ik * packC * BITS + offset_c * BITS + 0]));

          __m256i va2 = _mm256_loadu_si256((__m256i*) (&buffer[offset_c * BITS + 4]));
          __m256i vb2 = _mm256_loadu_si256((__m256i*) (&qxptr[ik * packC * BITS + offset_c * BITS + 4]));

          __m256i va3 = _mm256_unpacklo_epi64(va1, va2);
          __m256i vb3 = _mm256_unpacklo_epi64(vb1, vb2);

          __m256i va4 = _mm256_unpackhi_epi64(va1, va2);
          __m256i vb4 = _mm256_unpackhi_epi64(vb1, vb2);

          __m256i p1 = _mm256_xor_si256(va3, vb3);
          __m256i p2 = _mm256_and_si256(va4, vb4);
          __m256i p3 = _mm256_and_si256(p1, p2);

          _mm256_storeu_si256((__m256i*) &popcnt_buffer1[ic / 2], p3);
          _mm256_storeu_si256((__m256i*) &popcnt_buffer2[ic / 2], p2);

          for (int64_t i = ic + 4; i < ic + 8; i++) {
            offset_c = bc + i;
            int64_t x1 = qxptr[ik * packC * BITS + offset_c * BITS + 0];
            int64_t x2 = qxptr[ik * packC * BITS + offset_c * BITS + 1];

            int64_t p11 = buffer[offset_c * BITS + 0] ^ x1;
            int64_t p12 = buffer[offset_c * BITS + 1] & x2;

            int64_t p13 = p11 & p12;

            cntp1 = cntp1 + popcnt64(p13);
            cntp2 = cntp2 + popcnt64(p12);
          }
        }

        int64_t ic_b = ic;
        for (; ic < block_size; ic++) {
          int64_t x1 = qxptr[ik * packC * BITS + ic * BITS + 0];
          int64_t x2 = qxptr[ik * packC * BITS + ic * BITS + 1];

          int64_t p1 = buffer[ic * BITS + 0] ^ x1;
          int64_t p2 = buffer[ic * BITS + 1] & x2;

          int64_t p3 = p1 & p2;

          cntp1 = cntp1 + popcnt64(p3);
          cntp2 = cntp2 + popcnt64(p2);
        }

        cntp1 += popcnt(popcnt_buffer1.data(), (ic_b / 2) * 8);
        cntp2 += popcnt(popcnt_buffer2.data(), (ic_b / 2) * 8);
      }

      int temp = cntp2 - cntp1 - cntp1;

      if (temp > 0) {
        res[ik * num_kernels + in] = temp;
      } else {
        res[ik * num_kernels + in] = temp * relu_alpha;
      }
    }
  }
}