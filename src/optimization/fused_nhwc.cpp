#include "fused_nhwc.h"

/*
  General version for quantizing N x H x W x C to N x H x W x C_p x BITS.
  Ressembles quantize version for N x C x H x W but has better locality
  since input matches output shape and quantization is done on C.
*/
std::vector<int64_t> quantize_NHWC_to_NHWCB(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int64_t priChannel = C / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  // The quantized qx, in N_H_W_C_B format
  std::vector<int64_t> qx = std::vector<int64_t>(N * packH * packW * packC * BITS, 0);
  int64_t* qxptr = qx.data();

  for (int64_t in = 0; in < N; in++) {
    for (int64_t ih = 0; ih < H; ih++) {
      for (int64_t iw = 0; iw < W; iw++) {
        // Pack the first part: 0 ~ priChannel*cntbits
        for (int64_t ic = 0; ic < priChannel; ic++) {
          // for 2-bit packing
          int64_t p1 = 0;
          int64_t p2 = 0;
          for (int64_t bit = 0; bit < cntbits; bit++) {
            float currentx = X[((((in * H) + ih) * W) + iw) * C + ic * cntbits + bit];
            if (currentx > Q_Threshold) {
              // Pack 1: 01
              p2 = p2 | onebit[bit];
            } else if (currentx < (-Q_Threshold)) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }
          
          // Store the ternarized and packed data in N_H_W_C_B format
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + ic) * BITS + 0] = p1;
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + ic) * BITS + 1] = p2;
        }

        // Pack the second part: priChannel*cntbits ~ C
        if ((C % cntbits) > 0) {
          int64_t p1 = 0;
          int64_t p2 = 0;
          for (int64_t bit = 0; bit < (C % cntbits); bit++) {
            float currentx = X[((((in * H) + ih) * W) + iw) * C + priChannel * cntbits + bit];
            if (currentx > Q_Threshold) {
              // Pack 1: 01
              p2 = p2 | onebit[bit];
            } else if (currentx < (-Q_Threshold)) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }

          // Store packed data into new NHWCB format
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + priChannel) * BITS + 0] = p1;
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + priChannel) * BITS + 1] = p2;
        }
      }
    }
  }

  return qx;
}

/*
  SIMD version for quantizing N x H x W x C to N x H x W x C_p x BITS.
  Applies SIMD on the bit loop.
*/
std::vector<int64_t> quantize_NHWC_to_NHWCB_simd(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int64_t priChannel = C / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  // The quantized qx, in N_H_W_C_B format
  std::vector<int64_t> qx = std::vector<int64_t>(N * packH * packW * packC * BITS, 0);
  int64_t* qxptr = qx.data();

  for (int64_t in = 0; in < N; in++) {
    for (int64_t ih = 0; ih < H; ih++) {
      for (int64_t iw = 0; iw < W; iw++) {
        for (int64_t ic = 0; ic < priChannel; ic++) {
          // for 2-bit packing of channel
          int64_t p1 = 0;
          int64_t p2 = 0;

          const __m256 threshold = _mm256_set1_ps(Q_Threshold);
          const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);
          for (int64_t bit = 0; bit < cntbits; bit += 8) {
            __m256 currentx = _mm256_loadu_ps(X + ((((in * H) + ih) * W) + iw) * C + ic * cntbits + bit); // for general C not aligned

            __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
            __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

            int64_t gt_mask = _mm256_movemask_ps(mask_gt);
            int64_t lt_mask = _mm256_movemask_ps(mask_lt);

            p1 |= (lt_mask << bit);
            p2 |= (gt_mask << bit);
            p2 |= (lt_mask << bit);
          }

          // Store the ternarized and packed data in N_H_W_C_B format
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + ic) * BITS + 0] = p1;
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + ic) * BITS + 1] = p2;
        }

        // Pack the second part: priChannel*cntbits ~ C
        if ((C % cntbits) > 0) {
          int64_t p1 = 0;
          int64_t p2 = 0;

          const __m256 threshold = _mm256_set1_ps(Q_Threshold);
          const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);

          int64_t bit = 0;
          for (; bit <= (C % cntbits) - 8; bit += 8) {
            __m256 currentx = _mm256_loadu_ps(X + ((((in * H) + ih) * W) + iw) * C + priChannel * cntbits + bit);

            __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
            __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

            int64_t gt_mask = _mm256_movemask_ps(mask_gt);
            int64_t lt_mask = _mm256_movemask_ps(mask_lt);

            p1 |= (lt_mask << bit);
            p2 |= (gt_mask << bit);
            p2 |= (lt_mask << bit);
          }
          for (; bit < (C % cntbits); bit++) {
            float currentx = X[((((in * H) + ih) * W) + iw) * C + priChannel * cntbits + bit];
            if (currentx > Q_Threshold) {
              // Pack 1: 01
              p2 = p2 | onebit[bit];
            } else if (currentx < (-Q_Threshold)) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }

          // Store packed data into new NHWCB format
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + priChannel) * BITS + 0] = p1;
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + priChannel) * BITS + 1] = p2;
        }
      }
    }
  }

  return qx;
}


/*
  First version for fusing quantization and img2row for the format N x H x W x C to N x H x W x C_p x BITS.
  The function first performs quantization, then immediately copies the data to the (at most KH * KW)
  locations in the output, where the values are needed.
*/
std::vector<int64_t> fused_NHWC_to_NHWCB(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int64_t priChannel = C / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  const int64_t OH = (packH - KH) / stride + 1;
  const int64_t OW = (packW - KW) / stride + 1;
  const int64_t H1 = OH * OW;                 // Fused Height
  const int64_t W1 = KH * KW * packC * BITS;  // Fused Width

  // The quantized qx, in N_H_W_C_B format
  std::vector<int64_t> qx = std::vector<int64_t>(N * H1 * W1, 0);
  int64_t* qxptr = qx.data();

  for (int64_t in = 0; in < N; in++) {
    for (int64_t ih = 0; ih < H; ih++) {
      for (int64_t iw = 0; iw < W; iw++) {
        // Pack the first part: 0 ~ priChannel*cntbits
        for (int64_t ic = 0; ic < priChannel; ic++) {
          // for 2-bit packing of channel
          int64_t p1 = 0;
          int64_t p2 = 0;
          for (int64_t bit = 0; bit < cntbits; bit++) {
            float currentx = X[((((in * H) + ih) * W) + iw) * C + ic * cntbits + bit];
            if (currentx > Q_Threshold) {
              // Pack 1: 01
              p2 = p2 | onebit[bit];
            } else if (currentx < (-Q_Threshold)) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }

          int64_t padded_h = ih + padding;
          int64_t padded_w = iw + padding;

          for (int64_t kh = 0; kh < KH; kh++) {
            for (int64_t kw = 0; kw < KW; kw++) {
              int64_t out_h = (padded_h - kh) / stride;
              int64_t out_w = (padded_w - kw) / stride;

              // Check if the current position falls within the valid output dimensions
              if (out_h >= 0 && out_h < OH && out_w >= 0 && out_w < OW 
                  && (padded_h - kh) % stride == 0 && (padded_w - kw) % stride == 0) {
                int64_t output_row = in * OH * OW + out_h * OW + out_w;
                int64_t output_col = (kh * KW + kw) * packC * BITS + ic * BITS;

                qxptr[output_row * W1 + output_col + 0] = p1;
                qxptr[output_row * W1 + output_col + 1] = p2;
              }
            }
          }
        }

        // Pack the second part: priChannel*cntbits ~ C
        if ((C % cntbits) > 0) {
          int64_t p1 = 0;
          int64_t p2 = 0;
          for (int64_t bit = 0; bit < (C % cntbits); bit++) {
            float currentx = X[((((in * H) + ih) * W) + iw) * C + priChannel * cntbits + bit];
            if (currentx > Q_Threshold) {
              // Pack 1: 01
              p2 = p2 | onebit[bit];
            } else if (currentx < (-Q_Threshold)) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }

          int64_t padded_h = ih + padding;
          int64_t padded_w = iw + padding;

          for (int64_t kh = 0; kh < KH; kh++) {
            for (int64_t kw = 0; kw < KW; kw++) {
              int64_t out_h = (padded_h - kh) / stride;
              int64_t out_w = (padded_w - kw) / stride;

              // Check if the current position falls within the valid output dimensions
              if (out_h >= 0 && out_h < OH && out_w >= 0 && out_w < OW 
                  && (padded_h - kh) % stride == 0 && (padded_w - kw) % stride == 0) {
                int64_t output_row = in * OH * OW + out_h * OW + out_w;
                int64_t output_col = (kh * KW + kw) * packC * BITS + priChannel * BITS;

                qxptr[output_row * W1 + output_col + 0] = p1;
                qxptr[output_row * W1 + output_col + 1] = p2;
              }
            }
          }
        }
      }
    }
  }

  return qx;
}

/*
  Same as first version for fusing quantization and img2row for the format N x H x W x C to N x H x W x C_p x BITS.
  The function first performs quantization, then immediately copies the data to the (at most KH * KW)
  locations in the output, where the values are needed. Only difference is the application of SIMD for packing.
*/
std::vector<int64_t> fused_NHWC_to_NHWCB_simd(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int64_t priChannel = C / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  const int64_t OH = (packH - KH) / stride + 1;
  const int64_t OW = (packW - KW) / stride + 1;
  const int64_t H1 = OH * OW;                 // Fused Height
  const int64_t W1 = KH * KW * packC * BITS;  // Fused Width

  // The quantized qx, in N_H_W_C_B format
  std::vector<int64_t> qx = std::vector<int64_t>(N * H1 * W1, 0);
  int64_t* qxptr = qx.data();

  for (int64_t in = 0; in < N; in++) {
    for (int64_t ih = 0; ih < H; ih++) {
      for (int64_t iw = 0; iw < W; iw++) {
        // Pack the first part: 0 ~ priChannel*cntbits
        for (int64_t ic = 0; ic < priChannel; ic++) {
          // for 2-bit packing of channel
          int64_t p1 = 0;
          int64_t p2 = 0;

          const __m256 threshold = _mm256_set1_ps(Q_Threshold);
          const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);
          for (int64_t bit = 0; bit < cntbits; bit += 8) {
            __m256 currentx = _mm256_loadu_ps(X + ((((in * H) + ih) * W) + iw) * C + ic * cntbits + bit); // for general C not aligned

            __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
            __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

            int64_t gt_mask = _mm256_movemask_ps(mask_gt);
            int64_t lt_mask = _mm256_movemask_ps(mask_lt);

            p1 |= (lt_mask << bit);
            p2 |= (gt_mask << bit);
            p2 |= (lt_mask << bit);
          }

          int64_t padded_h = ih + padding;
          int64_t padded_w = iw + padding;

          for (int64_t kh = 0; kh < KH; kh++) {
            for (int64_t kw = 0; kw < KW; kw++) {
              int64_t out_h = (padded_h - kh) / stride;
              int64_t out_w = (padded_w - kw) / stride;

              // Check if the current position falls within the valid output dimensions
              if (out_h >= 0 && out_h < OH && out_w >= 0 && out_w < OW 
                  && (padded_h - kh) % stride == 0 && (padded_w - kw) % stride == 0) {
                int64_t output_row = in * OH * OW + out_h * OW + out_w;
                int64_t output_col = (kh * KW + kw) * packC * BITS + ic * BITS;

                qxptr[output_row * W1 + output_col + 0] = p1;
                qxptr[output_row * W1 + output_col + 1] = p2;
              }
            }
          }
        }

        // Pack the second part: priChannel*cntbits ~ C
        if ((C % cntbits) > 0) {
          int64_t p1 = 0;
          int64_t p2 = 0;

          const __m256 threshold = _mm256_set1_ps(Q_Threshold);
          const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);

          int64_t bit = 0;
          for (; bit <= (C % cntbits) - 8; bit += 8) {
            __m256 currentx = _mm256_loadu_ps(X + ((((in * H) + ih) * W) + iw) * C + priChannel * cntbits + bit);

            __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
            __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

            int64_t gt_mask = _mm256_movemask_ps(mask_gt);
            int64_t lt_mask = _mm256_movemask_ps(mask_lt);

            p1 |= (lt_mask << bit);
            p2 |= (gt_mask << bit);
            p2 |= (lt_mask << bit);
          }
          for (; bit < (C % cntbits); bit++) {
            float currentx = X[((((in * H) + ih) * W) + iw) * C + priChannel * cntbits + bit];
            if (currentx > Q_Threshold) {
              // Pack 1: 01
              p2 = p2 | onebit[bit];
            } else if (currentx < (-Q_Threshold)) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }

          int64_t padded_h = ih + padding;
          int64_t padded_w = iw + padding;

          for (int64_t kh = 0; kh < KH; kh++) {
            for (int64_t kw = 0; kw < KW; kw++) {
              int64_t out_h = (padded_h - kh) / stride;
              int64_t out_w = (padded_w - kw) / stride;

              // Check if the current position falls within the valid output dimensions
              if (out_h >= 0 && out_h < OH && out_w >= 0 && out_w < OW 
                  && (padded_h - kh) % stride == 0 && (padded_w - kw) % stride == 0) {
                int64_t output_row = in * OH * OW + out_h * OW + out_w;
                int64_t output_col = (kh * KW + kw) * packC * BITS + priChannel * BITS;

                qxptr[output_row * W1 + output_col + 0] = p1;
                qxptr[output_row * W1 + output_col + 1] = p2;
              }
            }
          }
        }
      }
    }
  }

  return qx;
}

/*
  Second version for fusing quantization and img2row for the format N x H x W x C to N x H x W x C_p x BITS.
  The function first performs quantization, before aggregating the packed values for the different channels
  in a buffer. Since these values will also be contigous in the output, a larger chunk can be directly copied
  to the output.
*/
std::vector<int64_t> fused_NHWC_to_NHWCB_buffer(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int64_t priChannel = C / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  const int64_t OH = (packH - KH) / stride + 1;
  const int64_t OW = (packW - KW) / stride + 1;
  const int64_t H1 = OH * OW;                 // Fused Height
  const int64_t W1 = KH * KW * packC * BITS;  // Fused Width

  // The quantized qx, in N_H_W_C_B format
  std::vector<int64_t> qx = std::vector<int64_t>(N * H1 * W1, 0);
  int64_t* qxptr = qx.data();

  for (int64_t in = 0; in < N; in++) {
    for (int64_t ih = 0; ih < H; ih++) {
      for (int64_t iw = 0; iw < W; iw++) {
        // Pack the first part: 0 ~ priChannel*cntbits
        std::vector<int64_t> buffer = std::vector<int64_t>(packC * BITS, 0);
        for (int64_t ic = 0; ic < priChannel; ic++) {
          // for 2-bit packing of channel
          int64_t p1 = 0;
          int64_t p2 = 0;

          const __m256 threshold = _mm256_set1_ps(Q_Threshold);
          const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);
          for (int64_t bit = 0; bit < cntbits; bit += 8) {
            __m256 currentx = _mm256_loadu_ps(X + ((((in * H) + ih) * W) + iw) * C + ic * cntbits + bit); // for general C not aligned

            __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
            __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

            int64_t gt_mask = _mm256_movemask_ps(mask_gt);
            int64_t lt_mask = _mm256_movemask_ps(mask_lt);

            p1 |= (lt_mask << bit);
            p2 |= (gt_mask << bit);
            p2 |= (lt_mask << bit);
          }

          buffer[ic * BITS + 0] = p1;
          buffer[ic * BITS + 1] = p2;
        }

        // Pack the second part: priChannel*cntbits ~ C
        if ((C % cntbits) > 0) {
          int64_t p1 = 0;
          int64_t p2 = 0;

          const __m256 threshold = _mm256_set1_ps(Q_Threshold);
          const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);

          int64_t bit = 0;
          for (; bit <= (C % cntbits) - 8; bit += 8) {
            __m256 currentx = _mm256_loadu_ps(X + ((((in * H) + ih) * W) + iw) * C + priChannel * cntbits + bit);

            __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
            __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

            int64_t gt_mask = _mm256_movemask_ps(mask_gt);
            int64_t lt_mask = _mm256_movemask_ps(mask_lt);

            p1 |= (lt_mask << bit);
            p2 |= (gt_mask << bit);
            p2 |= (lt_mask << bit);
          }
          for (; bit < (C % cntbits); bit++) {
            float currentx = X[((((in * H) + ih) * W) + iw) * C + priChannel * cntbits + bit];
            if (currentx > Q_Threshold) {
              // Pack 1: 01
              p2 = p2 | onebit[bit];
            } else if (currentx < (-Q_Threshold)) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }

          buffer[priChannel * BITS + 0] = p1;
          buffer[priChannel * BITS + 1] = p2;
        }

        int64_t padded_h = ih + padding;
        int64_t padded_w = iw + padding;

        // Calculate lower bounds
        int64_t kh_start = (padded_h % stride == 0) ? 0 : stride - (padded_h % stride);
        int64_t kw_start = (padded_w % stride == 0) ? 0 : stride - (padded_w % stride);

        for (int64_t kh = kh_start; kh < KH; kh += stride) {
          for (int64_t kw = kw_start; kw < KW; kw += stride) {
            int64_t out_h = (padded_h - kh) / stride;
            int64_t out_w = (padded_w - kw) / stride;

            // could infer (tighter) upper and lower bounds for the for-loop
            // but there are issues with int arithmetic.
            // see that     x = y / z    =/=>    x * z = y    for general integers x, y, z 
            if (out_h >= 0 && out_w >= 0 && out_h < OH && out_w < OW) {
              int64_t output_row = in * OH * OW + out_h * OW + out_w;
              int64_t output_col = kh * KW * packC * BITS + kw * packC * BITS;

              std::memcpy(&qxptr[output_row * W1 + output_col + 0], buffer.data(), packC * BITS * sizeof(int64_t));
            }
          }
        }
      }
    }
  }

  return qx;
}

/*
  Third version for fusing quantization and img2row for the format N x H x W x C to N x H x W x C_p x BITS.
  This version builds upon the second one but uses a larger buffer. We observe that most contiguous values in X
  also appear as contiguous values in the output (usually of size KW). This fact is exploited by blocking on
  height/width and by allocating a larger buffer. Then, the buffer is first filled with the packed values
  for a particular block. Afterwards, the contigous regions are copied from the buffer to the correct locations
  in the output.

  Attention: This method assumes stride = 1
*/
std::vector<int64_t> fused_NHWC_to_NHWCB_blocked(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride) {
  const int64_t one = 1;
  int64_t onebit[cntbits];

  // find out best block size for particular input.
  const int64_t block_size_h = KH;
  const int64_t block_size_w = W;
  
  // 64-bits, set each bit
  for (int i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int64_t priChannel = C / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  const int64_t OH = (packH - KH) / stride + 1;
  const int64_t OW = (packW - KW) / stride + 1;
  const int64_t H1 = N * OH * OW;             // Fused Height
  const int64_t W1 = KH * KW * packC * BITS;  // Fused Width

  // The quantized qx, in N_H_W_C_B format
  std::vector<int64_t> qx = std::vector<int64_t>(H1 * W1, 0);
  int64_t* qxptr = qx.data();

  // allocate buffer
  std::vector<int64_t> buffer = std::vector<int64_t>(block_size_h * block_size_w * packC * BITS);

  for (int64_t in = 0; in < N; in++) {
    for (int64_t ih = 0; ih < H; ih += block_size_h) {
      for (int64_t iw = 0; iw < W; iw += block_size_w) {
        int64_t bs_h = std::min(H - ih, block_size_h);
        int64_t bs_w = std::min(W - iw, block_size_w);

        for (int64_t i = 0; i < bs_h; i++) {
          for (int64_t j = 0; j < bs_w; j++) {
            int64_t bh = ih + i;
            int64_t bw = iw + j;

            for (int64_t ic = 0; ic < priChannel; ic++) {
              // for 2-bit packing of channel
              int64_t p1 = 0;
              int64_t p2 = 0;

              const __m256 threshold = _mm256_set1_ps(Q_Threshold);
              const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);
              for (int64_t bit = 0; bit < cntbits; bit += 8) {
                __m256 currentx = _mm256_loadu_ps(X + ((((in * H) + bh) * W) + bw) * C + ic * cntbits + bit);

                __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
                __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

                int64_t gt_mask = _mm256_movemask_ps(mask_gt);
                int64_t lt_mask = _mm256_movemask_ps(mask_lt);

                p1 |= (lt_mask << bit);
                p2 |= (gt_mask << bit);
                p2 |= (lt_mask << bit);
              }

              buffer[((i * bs_w + j) * packC + ic) * BITS + 0] = p1;
              buffer[((i * bs_w + j) * packC + ic) * BITS + 1] = p2;
            }

            // Pack the second part: priChannel*cntbits ~ C
            if ((C % cntbits) > 0) {
              int64_t p1 = 0;
              int64_t p2 = 0;

              const __m256 threshold = _mm256_set1_ps(Q_Threshold);
              const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);

              int64_t bit = 0;
              for (; bit <= (C % cntbits) - 8; bit += 8) {
                __m256 currentx = _mm256_loadu_ps(X + ((((in * H) + bh) * W) + bw) * C + priChannel * cntbits + bit);

                __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
                __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

                int64_t gt_mask = _mm256_movemask_ps(mask_gt);
                int64_t lt_mask = _mm256_movemask_ps(mask_lt);

                p1 |= (lt_mask << bit);
                p2 |= (gt_mask << bit);
                p2 |= (lt_mask << bit);
              }
              for (; bit < (C % cntbits); bit++) {
                float currentx = X[((((in * H) + bh) * W) + bw) * C + priChannel * cntbits + bit];
                if (currentx > Q_Threshold) {
                  // Pack 1: 01
                  p2 = p2 | onebit[bit];
                } else if (currentx < (-Q_Threshold)) {
                  // Pack -1: 11
                  p1 = p1 | onebit[bit];
                  p2 = p2 | onebit[bit];
                }
              }

              buffer[((i * bs_w + j) * packC + priChannel) * BITS + 0] = p1;
              buffer[((i * bs_w + j) * packC + priChannel) * BITS + 1] = p2;
            }
          }
        }

        int64_t begin_h = ih + padding;  // range: [padded_h, padded_h + bs_h)
        int64_t begin_w = iw + padding;  // range: [padded_w, padded_w + bs_w)

        int64_t min_c = std::max(begin_h - KH + 1, (int64_t) 0);
        int64_t min_d = std::max(begin_w - KW + 1, (int64_t) 0);

        int64_t max_c = std::min(begin_h + bs_h, OH);
        int64_t max_d = std::min(begin_w + bs_w, OW);

        for (int64_t c = min_c; c < max_c; c++) {
          for (int64_t d = min_d; d < max_d; d++) {
            // for all x, y where
            // padded_h + x - kh = c
            // padded_w + y - kw = d
            // => the elements padded_h + x and padded_h + y are written to same row: c * OW + d
            // => the column for an element is: kh * KW + kw

            int64_t output_row = in * OH * OW + c * OW + d;

            int64_t kh_start = std::max(begin_h - c, (int64_t) 0);
            int64_t kh_end = std::min(KH, bs_h + begin_h - c);

            int64_t kw_start = std::max(begin_w - d, (int64_t) 0);
            int64_t kw_end = std::min(KW, bs_w + begin_w - d);
            int64_t kw_range = kw_end - kw_start;

            for (int64_t kh = kh_start; kh < kh_end; kh++) {
              int64_t output_col = (kh * KW + kw_start) * packC * BITS;

              int64_t x = c + kh - begin_h;
              int64_t y = d + kw_start - begin_w;

              // can copy a larger chunk from buffer to output
              std::memcpy(&qxptr[output_row * W1 + output_col + 0], &buffer[(x * bs_w + y) * packC * BITS], kw_range * packC * BITS * sizeof(int64_t));
            }
          }
        }
      }
    }
  }
  return qx;
}

static void inline fill_buffer(float* X, int64_t* buffer, int64_t* onebit, int64_t C, int64_t H, int64_t W, int64_t ih, int64_t i, int64_t in, int64_t packC, int64_t priChannel, float Q_Threshold) {
  for (int64_t j = 0; j < W; j++) {
    int64_t bh = ih + i;

    for (int64_t ic = 0; ic < priChannel; ic++) {
      // for 2-bit packing of channel
      int64_t p1 = 0;
      int64_t p2 = 0;

      const __m256 threshold = _mm256_set1_ps(Q_Threshold);
      const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);
      for (int64_t bit = 0; bit < cntbits; bit += 8) {
        __m256 currentx = _mm256_loadu_ps(X + ((((in * H) + bh) * W) + j) * C + ic * cntbits + bit);

        __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
        __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

        int64_t gt_mask = _mm256_movemask_ps(mask_gt);
        int64_t lt_mask = _mm256_movemask_ps(mask_lt);

        p1 |= (lt_mask << bit);
        p2 |= (gt_mask << bit);
        p2 |= (lt_mask << bit);
      }

      buffer[((i * W + j) * packC + ic) * BITS + 0] = p1;
      buffer[((i * W + j) * packC + ic) * BITS + 1] = p2;
    }

    // Pack the second part: priChannel*cntbits ~ C
    if ((C % cntbits) > 0) {
      int64_t p1 = 0;
      int64_t p2 = 0;

      const __m256 threshold = _mm256_set1_ps(Q_Threshold);
      const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);

      int64_t bit = 0;
      for (; bit <= (C % cntbits) - 8; bit += 8) {
        __m256 currentx = _mm256_loadu_ps(X + ((((in * H) + bh) * W) + j) * C + priChannel * cntbits + bit);

        __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
        __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

        int64_t gt_mask = _mm256_movemask_ps(mask_gt);
        int64_t lt_mask = _mm256_movemask_ps(mask_lt);

        p1 |= (lt_mask << bit);
        p2 |= (gt_mask << bit);
        p2 |= (lt_mask << bit);
      }
      for (; bit < (C % cntbits); bit++) {
        float currentx = X[((((in * H) + bh) * W) + j) * C + priChannel * cntbits + bit];
        if (currentx > Q_Threshold) {
          // Pack 1: 01
          p2 = p2 | onebit[bit];
        } else if (currentx < (-Q_Threshold)) {
          // Pack -1: 11
          p1 = p1 | onebit[bit];
          p2 = p2 | onebit[bit];
        }
      }

      buffer[((i * W + j) * packC + priChannel) * BITS + 0] = p1;
      buffer[((i * W + j) * packC + priChannel) * BITS + 1] = p2;
    }
  }
}


/*
  This is the 3x3 kernel version for fusing quantization and img2row for the 
  format N x H x W x C to N x H x W x C_p x BITS. It takes the 3rd version as a starting point
  and applies optimization techniques that are applicable for this particular case.

  Attention: Assumes W >= 2, H >= 2, block_size_h <= KH, padding = 1, stride = 1, KW = KH = 3
*/
std::vector<int64_t> fused_NHWC_to_NHWCB_blocked3x3(float* X, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W) {
  const int64_t KH = 3;
  const int64_t KW = 3;

  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // find out best block size for particular input.
  const int64_t block_size_h = KH;

  // initial packed channel num
  const int64_t priChannel = C / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;

  const int64_t H1 = N * H * W;               // Fused Height
  const int64_t W1 = KH * KW * packC * BITS;  // Fused Width

  // The quantized qx, in N_H_W_C_B format
  std::vector<int64_t> qx = std::vector<int64_t>(H1 * W1, 0);
  int64_t* qxptr = qx.data();

  // allocate buffer
  std::vector<int64_t> buffer = std::vector<int64_t>(block_size_h * W * packC * BITS);

  for (int64_t in = 0; in < N; in++) {
    // ih = 0
    fill_buffer(X, buffer.data(), onebit, C, H, W, 0, 0, in, packC, priChannel, Q_Threshold);

    // c = 0
    // d = 0
    std::memcpy(&qxptr[in * H * W * W1 + (KW + 1) * packC * BITS], buffer.data(), 2 * packC * BITS * sizeof(int64_t));

    // 1 < d < W - 1
    for (int64_t d = 0; d < W - 2; d++) {
      const int64_t output_row = in * H * W + d + 1;
      std::memcpy(&qxptr[output_row * W1 + KW * packC * BITS], &buffer[d * packC * BITS], 3 * packC * BITS * sizeof(int64_t));
    }

    // d = W - 1
    std::memcpy(&qxptr[(in * H * W + W - 1) * W1 + KW * packC * BITS], &buffer[(W - 2) * packC * BITS], 2 * packC * BITS * sizeof(int64_t));

    // c = 1
    // d = 0
    std::memcpy(&qxptr[(in * H * W + W) * W1 + packC * BITS], buffer.data(), 2 * packC * BITS * sizeof(int64_t));

    // 1 < d < W - 1
    for (int64_t d = 0; d < W - 2; d++) {
      const int64_t output_row = in * H * W + W + d + 1;
      std::memcpy(&qxptr[output_row * W1], &buffer[d * packC * BITS], 3 * packC * BITS * sizeof(int64_t));
    }

    // d = W - 1
    std::memcpy(&qxptr[(in * H * W + W + W - 1) * W1], &buffer[(W - 2) * packC * BITS], 2 * packC * BITS * sizeof(int64_t));

    // 0 < ih < H - 1
    for (int64_t ih = 1; ih < H - 1; ih += block_size_h) {
      const int64_t bs_h = std::min(H - 1 - ih, block_size_h);

      for (int64_t i = 0; i < bs_h; i++) {
        fill_buffer(X, buffer.data(), onebit, C, H, W, ih, i, in, packC, priChannel, Q_Threshold);
      }

      const int64_t begin_h = ih + 1;

      // c = begin_h - 2
      // d = 0
      const int64_t output_row = in * H * W + (begin_h - 2) * W;
      std::memcpy(&qxptr[output_row * W1 + (2 * KW + 1) * packC * BITS], buffer.data(), 2 * packC * BITS * sizeof(int64_t));

      // 1 < d < W - 1
      for (int64_t d = 0; d < W - 2; d++) {
        const int64_t output_row1 = in * H * W + (begin_h - 2) * W + d + 1;
        std::memcpy(&qxptr[output_row1 * W1 + 2 * KW * packC * BITS], &buffer[d * packC * BITS], 3 * packC * BITS * sizeof(int64_t));
      }

      // d = W - 1
      const int64_t output_row1 = in * H * W + (begin_h - 2) * W + W - 1;
      std::memcpy(&qxptr[output_row1 * W1 + 2 * KW * packC * BITS], &buffer[(W - 2) * packC * BITS], 2 * packC * BITS * sizeof(int64_t));

      // c = begin_h - 1
      // d = 0
      const int64_t output_row2 = in * H * W + (begin_h - 1) * W;
      int64_t kh_end = std::min(KH, bs_h + 1);

      for (int64_t kh = 1; kh < kh_end; kh++) {
        const int64_t output_col = (kh * KW + 1) * packC * BITS;

        std::memcpy(&qxptr[output_row2 * W1 + output_col], &buffer[(kh - 1) * W * packC * BITS], 2 * packC * BITS * sizeof(int64_t));
      }

      // 1 < d < W - 1
      for (int64_t d = 0; d < W - 2; d++) {
        const int64_t output_row3 = in * H * W + (begin_h - 1) * W + d + 1;

        for (int64_t kh = 1; kh < kh_end; kh++) {
          const int64_t output_col = kh * KW * packC * BITS;

          const int64_t x = (begin_h - 1) + kh - begin_h;

          std::memcpy(&qxptr[output_row3 * W1 + output_col], &buffer[(x * W + d) * packC * BITS], 3 * packC * BITS * sizeof(int64_t));
        }
      }

      // d = W - 1
      const int64_t output_row3 = in * H * W + (begin_h - 1) * W + W - 1;

      for (int64_t kh = 1; kh < kh_end; kh++) {
        const int64_t output_col = kh * KW * packC * BITS;

        const int64_t x = (begin_h - 1) + kh - begin_h;

        std::memcpy(&qxptr[output_row3 * W1 + output_col], &buffer[(x * W + W - 2) * packC * BITS], 2 * packC * BITS * sizeof(int64_t));
      }

      const int64_t max_c = begin_h + bs_h;  // <= H - 2
      for (int64_t c = begin_h; c < max_c; c++) {
        // d = 0
        const int64_t output_row4 = in * H * W + c * W;

        for (int64_t kh = 0; kh < bs_h; kh++) {
          const int64_t output_col = (kh * KW + 1) * packC * BITS;

          const int64_t x = c + kh - begin_h;

          std::memcpy(&qxptr[output_row4 * W1 + output_col], &buffer[x * W * packC * BITS], 2 * packC * BITS * sizeof(int64_t));
        }

        // 1 < d < W - 1
        for (int64_t d = 0; d < W - 2; d++) {
          const int64_t output_row5 = in * H * W + c * W + d + 1;

          for (int64_t kh = 0; kh < bs_h; kh++) {
            const int64_t output_col = kh * KW * packC * BITS;

            const int64_t x = c + kh - begin_h;

            std::memcpy(&qxptr[output_row5 * W1 + output_col], &buffer[(x * W + d) * packC * BITS], 3 * packC * BITS * sizeof(int64_t));
          }
        }

        // d = W - 1
        const int64_t output_row5 = in * H * W + c * W + W - 1;

        for (int64_t kh = 0; kh < bs_h; kh++) {
          const int64_t output_col = kh * KW * packC * BITS;

          const int64_t x = c + kh - begin_h;

          std::memcpy(&qxptr[output_row5 * W1 + output_col], &buffer[(x * W + W - 2) * packC * BITS], 2 * packC * BITS * sizeof(int64_t));
        }
      }
    }

    // ih = H - 1
    fill_buffer(X, buffer.data(), onebit, C, H, W, H - 1, 0, in, packC, priChannel, Q_Threshold);

    // c = H - 2
    // d = 0
    const int64_t output_row = in * H * W + (H - 2) * W;
    std::memcpy(&qxptr[output_row * W1 + (2 * KW + 1) * packC * BITS], buffer.data(), 2 * packC * BITS * sizeof(int64_t));

    // 1 < d < W - 1
    for (int64_t d = 0; d < W - 2; d++) {
      const int64_t output_row1 = in * H * W + (H - 2) * W + d + 1;
      std::memcpy(&qxptr[output_row1 * W1 + 2 * KW * packC * BITS], &buffer[d * packC * BITS], 3 * packC * BITS * sizeof(int64_t));
    }

    // d = W - 1
    const int64_t output_row1 = in * H * W + (H - 2) * W + W - 1;
    std::memcpy(&qxptr[output_row1 * W1 + 2 * KW * packC * BITS], &buffer[(W - 2) * packC * BITS], 2 * packC * BITS * sizeof(int64_t));

    // c = H - 1
    // d = 0
    const int64_t output_row2 = in * H * W + (H - 1) * W;
    std::memcpy(&qxptr[output_row2 * W1 + (KW + 1) * packC * BITS], buffer.data(), 2 * packC * BITS * sizeof(int64_t));

    // 1 < d < W - 1
    for (int64_t d = 0; d < W - 2; d++) {
      const int64_t output_row3 = in * H * W + (H - 1) * W + d + 1;
      std::memcpy(&qxptr[output_row3 * W1 + KW * packC * BITS], &buffer[d * packC * BITS], 3 * packC * BITS * sizeof(int64_t));
    }

    // d = W - 1
    const int64_t output_row3 = in * H * W + (H - 1) * W + W - 1;
    std::memcpy(&qxptr[output_row3 * W1 + KW * packC * BITS], &buffer[(W - 2) * packC * BITS], 2 * packC * BITS * sizeof(int64_t));
  }

  return qx;
}