#include "yuliia_lucas_functions.h"

/////////////////////////////////////////////////////////////////////////////////
//////////////////////        SIMD QUANTIZE-PACK        /////////////////////////
/////////////////////////////////////////////////////////////////////////////////

static inline void pack_SIMD(float* X, int64_t H, int64_t W, int64_t C, int64_t in, int64_t ic, int64_t ih, int64_t iw, float Q_Threshold, int64_t* onebit, int64_t& p1, int64_t& p2) {
  const __m256 threshold = _mm256_set1_ps(Q_Threshold);
  const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);
  for (int64_t bit = 0; bit < cntbits; bit += 8) {
    int64_t base_address = in * C * H * W + (ic * cntbits + bit) * H * W + ih * W + iw;
    __m256 currentx = _mm256_set_ps(X[base_address + 7 * H * W], X[base_address + 6 * H * W], X[base_address + 5 * H * W], X[base_address + 4 * H * W], X[base_address + 3 * H * W], X[base_address + 2 * H * W], X[base_address + 1 * H * W], X[base_address]);

    __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
    __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

    int64_t gt_mask = _mm256_movemask_ps(mask_gt);
    int64_t lt_mask = _mm256_movemask_ps(mask_lt);

    p1 |= (lt_mask << bit);
    p2 |= (gt_mask << bit);
    p2 |= (lt_mask << bit);
  }
}

// the overhanging part of the channel dimension needs to be handled separately when doing quantize-pack with SIMD
static inline void pack_SIMD_overhang(float* X, int64_t H, int64_t W, int64_t C, int64_t in, int64_t priChannel, int64_t ih, int64_t iw, float Q_Threshold, int64_t* onebit, int64_t& p1, int64_t& p2) {
  const __m256 threshold = _mm256_set1_ps(Q_Threshold);
  const __m256 neg_threshold = _mm256_set1_ps(-Q_Threshold);

  int64_t bit = 0;
  for (; bit <= (C % cntbits) - 8; bit += 8) {
    int64_t base_address = in * C * H * W + (priChannel * cntbits + bit) * H * W + ih * W + iw;
    __m256 currentx = _mm256_set_ps(X[base_address + 7 * H * W], X[base_address + 6 * H * W], X[base_address + 5 * H * W], X[base_address + 4 * H * W], X[base_address + 3 * H * W], X[base_address + 2 * H * W], X[base_address + 1 * H * W], X[base_address]);

    __m256 mask_gt = _mm256_cmp_ps(currentx, threshold, _CMP_GT_OS);
    __m256 mask_lt = _mm256_cmp_ps(currentx, neg_threshold, _CMP_LT_OS);

    int64_t gt_mask = _mm256_movemask_ps(mask_gt);
    int64_t lt_mask = _mm256_movemask_ps(mask_lt);

    p1 |= (lt_mask << bit);
    p2 |= (gt_mask << bit);
    p2 |= (lt_mask << bit);
  }
  // this part has to be done in a non-vectorized way
  for (; bit < (C % cntbits); bit++) {
    float currentx = X[((in * C + (priChannel * cntbits + bit)) * H + ih) * W + iw];
    if (currentx > Q_Threshold) {
      // Pack 1: 01
      p2 = p2 | onebit[bit];
    } else if (currentx < (-Q_Threshold)) {
      // Pack -1: 11
      p1 = p1 | onebit[bit];
      p2 = p2 | onebit[bit];
    }
  }
}

/*
  SIMD version for quantizing N x C x H x W to N x H x W x C_p x BITS.
  Applies SIMD on the bit loop.
*/
std::vector<int64_t> quantize_NCHW_to_NHWCB(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int64_t i = 0; i < cntbits; i++) {
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

          pack_SIMD(X, H, W, C, in, ic, ih, iw, Q_Threshold, onebit, p1, p2);

          // Store the ternarized and packed data in N_H_W_C_B format
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + ic) * BITS + 0] = p1;
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + ic) * BITS + 1] = p2;
        }

        // Pack the second part: priChannel*cntbits ~ C
        if ((C % cntbits) > 0) {
          int64_t p1 = 0;
          int64_t p2 = 0;

          pack_SIMD_overhang(X, H, W, C, in, priChannel, ih, iw, Q_Threshold, onebit, p1, p2);

          // Store packed data int64_to new NHWCB format
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + priChannel) * BITS + 0] = p1;
          qxptr[(((in * packH + ih + padding) * packW + iw + padding) * packC + priChannel) * BITS + 1] = p2;
        }
      }
    }
  }

  return qx;
}

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////        OUR FUSING        /////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

/*
REMARKS:

HIGH LEVEL IDEA
fuse quantization-packing function int64_to im2row function.
do it using the im2row loop order in order to make the writebacks of the fused code have good locality.
compute the needed packed values on the fly. this will lead to recomputations of the same value (e.g. E on the diag) but it might be worth it since (compared to the unfused code) we save 1 memory write and 1 memory read for each unique packed value AND because the computation of a packed value is not particularly expensive.
there are two alteranative ways to fuse:
- "Robins approach": compute packed values with normal_image loop order and then when a packed value is ready write it to all places in im2col matrix where it belongs (see E which gets placed onto entire diagonal in "flow" picture) (DRAWBACK: bad write locality (e.g. the writes of E along the diagonal are not consecutive in memory))
- paper pseudocode (page 13) (DRAWBACK: every packed value is written to memory twice: once in line 15 and then again in line 20)

IMPLEMENTATION DETAILS
the main trick was to derive formulas for h and w that depend on the loop indices from the im2row loop order s.t. we can compute the required packed value on the fly as we iterate over the im2row matrix in im2row loop order. We drived these formulas by inspecting the indexing of X in the im2col.h file and by correctly accounting for the padding (= the "- PaddingH" and " - PaddingW" terms). Then we inlined the computation of the packed value and used the loop index c (innermost loop) to determine ic from the innermost loop from Quantize.cpp (which loops until priChannel) which allowed to delete this loop.
The writeback int64_to the im2row matrix uses the same indexing as the im2row.h file (wich is desired/the whole point64_t of this indexing as this indexing has good write locality). In order for the fusion to be correct we need to write back a 0 in case the computed h or w exceed the normal_image bounds.
*/

// NCHW to NHWCB
//
// our fusing (= im2row loop order)
std::vector<int64_t> our_fusing_1(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t OH, int64_t OW) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  // 64-bits, set each bit
  for (int64_t i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }
  
  // The quantized qx, in N_H_W_C_B format
  const int64_t priChannel = C / cntbits;
  
  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  int64_t C_packed_bits = packC * BITS;

  const int64_t H1 = OH * OW;      // Fused Height
  const int64_t W1 = KH * KW * C_packed_bits;  // Fused Width
  
  std::vector<int64_t> qx = std::vector<int64_t>(N * H1 * W1);

  for (int64_t n = 0; n < N; n++) {
    for (int64_t oh = 0; oh < OH; oh++) {
      for (int64_t ow = 0; ow < OW; ow++) {
        for (int64_t kh = 0; kh < KH; kh++) {
          for (int64_t kw = 0; kw < KW; kw++) {
            for (int64_t c = 0; c < C_packed_bits; c++) {
              // compute inds
              int64_t h = oh * stride + kh - padding;
              int64_t w = ow * stride + kw - padding;

              // compute value
              int64_t p1 = 0;
              int64_t p2 = 0;

              // figure out if c indexes int64_to data produced by loop or by if condition
              if (h >= 0 && h < H && w >= 0 && w < W) {
                if (c >= priChannel * 2) {  // c indexes int64_to if condition data
                  for (int64_t bit = 0; bit < (C % cntbits); bit++) {
                    float currentx = X[((n * C + (priChannel * cntbits + bit)) * H + h) * W + w];
                    if (currentx > Q_Threshold) {
                      p2 = p2 | onebit[bit];
                    } else if (currentx < -Q_Threshold) {
                      p1 = p1 | onebit[bit];
                      p2 = p2 | onebit[bit];
                    }
                  }
                } else {  // c indexes int64_to loop data so we need to compute ic
                  int64_t ic = c / 2;
                  for (int64_t bit = 0; bit < cntbits; bit++) {
                    float currentx = X[((n * C + (ic * cntbits + bit)) * H + h ) * W + w];
                    if (currentx > Q_Threshold) {
                      p2 = p2 | onebit[bit];
                    } else if (currentx < -Q_Threshold) {
                      p1 = p1 | onebit[bit];
                      p2 = p2 | onebit[bit];
                    }
                  }
                }
              }

              // writeback 
              qx[(n * H1 + oh * OW + ow) * W1 + kh * KW * C_packed_bits + kw * C_packed_bits + c] = (c % 2 == 0) ? p1 : p2;
            }
          }
        }
      }
    }
  }
  
  return qx;
}

// NCHW to NHWCB
//
// our fusing (= im2row loop order)
//
// uses SIMD for quant_pack
std::vector<int64_t> our_fusing_2(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t OH, int64_t OW) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  // 64-bits, set each bit
  for (int64_t i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }
  
  // The quantized qx, in N_H_W_C_B format
  const int64_t priChannel = C / cntbits;
  
  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  int64_t C_packed_bits = packC * BITS;

  const int64_t H1 = OH * OW;      // Fused Height
  const int64_t W1 = KH * KW * C_packed_bits;  // Fused Width
  
  std::vector<int64_t> qx = std::vector<int64_t>(N * H1 * W1);

  for (int64_t n = 0; n < N; n++) {
    for (int64_t oh = 0; oh < OH; oh++) {
      for (int64_t ow = 0; ow < OW; ow++) {
        for (int64_t kh = 0; kh < KH; kh++) {
          for (int64_t kw = 0; kw < KW; kw++) {
            for (int64_t c = 0; c < C_packed_bits; c++) {
              // compute inds
              int64_t h = oh * stride + kh - padding;
              int64_t w = ow * stride + kw - padding;

              // compute value
              int64_t p1 = 0;
              int64_t p2 = 0;

              // figure out if c indexes int64_to data produced by loop or by if condition
              if (h >= 0 && h < H && w >= 0 && w < W) {
                if (c >= priChannel * 2) {  // c indexes int64_to if condition data
                  pack_SIMD_overhang(X, H, W, C, n, priChannel, h, w, Q_Threshold, onebit, p1, p2);
                } else {  // c indexes int64_to loop data so we need to compute ic
                  int64_t ic = c / 2;
                  pack_SIMD(X, H, W, C, n, ic, h, w, Q_Threshold, onebit, p1, p2);
                }
              }

              // writeback 
              qx[(n * H1 + oh * OW + ow) * W1 + kh * KW * C_packed_bits + kw * C_packed_bits + c] = (c % 2 == 0) ? p1 : p2;
            }
          }
        }
      }
    }
  }
  
  return qx;
}

// NCHW to NHWCB
//
// our fusing (= im2row loop order)
//
// uses SIMD for quant_pack
//
// contains basic opts such as:
// - pull computation of h and w out of c loop (as they dont depend on c)
// - index int64_to QThreshold once outside of entire computation (it is always the same even for all n)
// - unrolled innermost loop by 2. this allowed us to:
//   - remove certain ops like some mults and divs
//   - make use (i.e. store) both p1 and p2 in each iteration (berfore we only made use of either in one loop iteration)
//
// derived fitting upper and lower bounds for kh and kw which allowed us to remove if condition in 2nd-innermost loop
std::vector<int64_t> our_fusing_3(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t OH, int64_t OW) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  // 64-bits, set each bit
  for (int64_t i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }
  
  // The quantized qx, in N_H_W_C_B format
  const int64_t priChannel = C / cntbits;
  
  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  int64_t C_packed_bits = packC * BITS;

  const int64_t H1 = OH * OW;      // Fused Height
  const int64_t W1 = KH * KW * C_packed_bits;  // Fused Width
  
  std::vector<int64_t> qx = std::vector<int64_t>(N * H1 * W1);

  for (int64_t n = 0; n < N; n++) {
    for (int64_t oh = 0; oh < OH; oh++) {

      int64_t kh_lower = (padding - oh * stride) > 0 ? (padding - oh * stride) : 0;
      int64_t kh_upper = std::min(KH, H - oh * stride + padding);

      for (int64_t ow = 0; ow < OW; ow++) {

        int64_t kw_lower = (padding - ow * stride) > 0 ? (padding - ow * stride) : 0;
        int64_t kw_upper = std::min(KW, W - ow * stride + padding);

        for (int64_t kh = kh_lower; kh < kh_upper; kh++) {
          for (int64_t kw = kw_lower; kw < kw_upper; kw++) {
            // compute inds
            int64_t h = oh * stride + kh - padding;
            int64_t w = ow * stride + kw - padding;

            // compute value
            int64_t p1 = 0;
            int64_t p2 = 0;

            // figure out if c indexes int64_to data produced by loop or by if condition
            for (int64_t c = 0; c < packC; c++) {  // unroll by 2
              int64_t p1 = 0;
              int64_t p2 = 0;

              if (c >= priChannel) {  // c indexes int64_to if condition data
                pack_SIMD_overhang(X, H, W, C, n, c, h, w, Q_Threshold, onebit, p1, p2);
              } else {  // c indexes int64_to loop data
                pack_SIMD(X, H, W, C, n, c, h, w, Q_Threshold, onebit, p1, p2);
              }

              // writeback 
              qx[(n * H1 + oh * OW + ow) * W1 + kh * KW * C_packed_bits + kw * C_packed_bits + 2 * c] = p1;
              qx[(n * H1 + oh * OW + ow) * W1 + kh * KW * C_packed_bits + kw * C_packed_bits + 2 * c + 1] = p2;
            }
          }
        }
      }
    }
  }
  return qx;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////        PAPER FUSING        ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

/*
REMARKS: 

Done by Lucas and Yuliia after Robin had done his fusing. Took robins code as a starting point64_t. changed read indexing of X to how its done in original ternarize (because unlike robins code we get tensor in NCHW). changed "(kh * KW + kw) * packC * BITS + ic" to "(kh * KW + kw) * packC * BITS + ic * BITS" (i.e. add the * BITS at the end) in writeback of ic loop. one test case was still failing so we searched for edge cases and came up with sightly improved way of computing OH and OW:

int64_t OH = (packH - KH) / stride;
int64_t OW = (packW - KW) / stride;
if ((packH - KH) % stride == (packW - KW) % stride) {
    OH = (packH - KH) / stride + 1;
    OW = (packW - KW) / stride + 1;
}

*/

// NCHW to NHWCB
//
// paper fusing (= normal image loop order)
std::vector<int64_t> paper_fusing_1(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int64_t i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int64_t priChannel = C / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

    int64_t OH = (packH - KH) / stride;
    int64_t OW = (packW - KW) / stride;

    if ((packH - KH) % stride == (packW - KW) % stride) {
        OH = (packH - KH) / stride + 1;
        OW = (packW - KW) / stride + 1;
    }

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
            float currentx = X[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw];
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

              if (out_h >= 0 && out_h < OH && out_w >= 0 && out_w < OW && (padded_h - kh) % stride == 0 && (padded_w - kw) % stride == 0) {
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
            float currentx = X[((in * C + (priChannel * cntbits + bit)) * H + ih) * W + iw];
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

              if (out_h >= 0 && out_h < OH && out_w >= 0 && out_w < OW && (padded_h - kh) % stride == 0 && (padded_w - kw) % stride == 0) {
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

// NCHW to NHWCB
//
// paper fusing (= normal image loop order)
//
// uses SIMD for quant_pack
std::vector<int64_t> paper_fusing_2(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW) {
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int64_t i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int64_t priChannel = C / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  int64_t OH = (packH - KH) / stride;
  int64_t OW = (packW - KW) / stride;

  if ((packH - KH) % stride == (packW - KW) % stride) {
      OH = (packH - KH) / stride + 1;
      OW = (packW - KW) / stride + 1;
  }

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
          
          pack_SIMD(X, H, W, C, in, ic, ih, iw, Q_Threshold, onebit, p1, p2);

          int64_t padded_h = ih + padding;
          int64_t padded_w = iw + padding;

          for (int64_t kh = 0; kh < KH; kh++) {
            for (int64_t kw = 0; kw < KW; kw++) {
              int64_t out_h = (padded_h - kh) / stride;
              int64_t out_w = (padded_w - kw) / stride;

              if (out_h >= 0 && out_h < OH && out_w >= 0 && out_w < OW && (padded_h - kh) % stride == 0 && (padded_w - kw) % stride == 0) {
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
          
          pack_SIMD_overhang(X, H, W, C, in, priChannel, ih, iw, Q_Threshold, onebit, p1, p2);

          int64_t padded_h = ih + padding;
          int64_t padded_w = iw + padding;

          for (int64_t kh = 0; kh < KH; kh++) {
            for (int64_t kw = 0; kw < KW; kw++) {
              int64_t out_h = (padded_h - kh) / stride;
              int64_t out_w = (padded_w - kw) / stride;

              if (out_h >= 0 && out_h < OH && out_w >= 0 && out_w < OW && (padded_h - kh) % stride == 0 && (padded_w - kw) % stride == 0) {
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

// NCHW to NHWCB
//
// paper fusing (= normal image loop order)
//
// uses SIMD for quant_pack
//
// contains basic opts
//
// derived fitting upper and lower bounds for kh and kw which allowed us to remove if condition
// 
//Important: here we assume that stride = 1.
//In this version the first idea was to use a larger buffer + memcpy. However, after discussion it was decided that with our data layout that would not provide any benefit as we cannot do blocking for H & W. Thus, the data is written back yo qx right after bit-packing.

std::vector<int64_t> paper_fusing_3(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW) {
 bool hasOverhang = C % cntbits;
  
  const int64_t one = 1;
  int64_t onebit[cntbits];
  
  // 64-bits, set each bit
  for (int64_t i = 0; i < cntbits; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int64_t priChannel = C / cntbits;

  // packC: actual packed input channel
  const int64_t packC = (C % cntbits) ? (priChannel + 1) : priChannel;
  const int64_t packH = H + 2 * padding;
  const int64_t packW = W + 2 * padding;

  int64_t OH = (packH - KH) / stride;
  int64_t OW = (packW - KW) / stride;

  if ((packH - KH) % stride == (packW - KW) % stride) {
      OH = (packH - KH) / stride + 1;
      OW = (packW - KW) / stride + 1;
  }

  const int64_t H1 = OH * OW;                 // Fused Height
  const int64_t W1 = KH * KW * packC * BITS;  // Fused Width

  const int64_t buffer_len = packC * BITS;

  // The quantized qx, in N_H_W_C_B format
  std::vector<int64_t> qx = std::vector<int64_t>(N * H1 * W1, 0);
  int64_t* qxptr = qx.data();

  for (int64_t in = 0; in < N; in++) {
    for (int64_t ih = 0; ih < H; ih++) {
      for (int64_t iw = 0; iw < W; iw++) {
        // Pack the first part: 0 ~ priChannel*cntbits
        std::vector<int64_t> buffer = std::vector<int64_t>(buffer_len);
        for (int64_t ic = 0; ic < priChannel; ic++) {
          // for 2-bit packing of channel
          int64_t p1 = 0;
          int64_t p2 = 0;
          
          pack_SIMD(X, H, W, C, in, ic, ih, iw, Q_Threshold, onebit, p1, p2);
        
          int64_t padded_h = ih + padding;
          int64_t padded_w = iw + padding;

          int64_t kh_start = (padded_h - (OH - 1)) > 0 ? (padded_h - (OH - 1)) : 0;
          int64_t kh_end = KH > padded_h + 1 ? padded_h + 1 : KH;

          int64_t kw_start = (padded_w - (OW - 1)) > 0 ? (padded_w - (OW - 1)) : 0;
          int64_t kw_end = KW > padded_w + 1 ? padded_w + 1 : KW;

          for (int64_t kh = kh_start; kh < kh_end; kh ++) {
            for (int64_t kw = kw_start; kw < kw_end; kw ++) {
              int64_t out_h = (padded_h - kh) / stride;
              int64_t out_w = (padded_w - kw) / stride;

              int64_t output_row = in * OH * OW + out_h * OW + out_w;
              int64_t output_col = (kh * KW + kw) * packC * BITS + ic * BITS;

              qxptr[output_row * W1 + output_col + 0] = p1;
              qxptr[output_row * W1 + output_col + 1] = p2;
            }
          }
        }

        // Pack the second part: priChannel*cntbits ~ C
        if ((hasOverhang) > 0) {
          int64_t p1 = 0;
          int64_t p2 = 0;
          
          pack_SIMD_overhang(X, H, W, C, in, priChannel, ih, iw, Q_Threshold, onebit, p1, p2);
          buffer[priChannel * BITS + 0] = p1;
          buffer[priChannel * BITS + 1] = p2;


          int64_t padded_h = ih + padding;
          int64_t padded_w = iw + padding;

          int64_t kh_start = (padded_h - (OH - 1)) > 0 ? (padded_h - (OH - 1)) : 0;
          int64_t kh_end = KH > padded_h + 1 ? padded_h + 1 : KH;

          int64_t kw_start = (padded_w - (OW - 1)) > 0 ? (padded_w - (OW - 1)) : 0;
          int64_t kw_end = KW > padded_w + 1 ? padded_w + 1 : KW;

          for (int64_t kh = kh_start; kh < kh_end; kh ++) {
            for (int64_t kw = kw_start; kw < kw_end; kw ++) {
              int64_t out_h = (padded_h - kh) / stride;
              int64_t out_w = (padded_w - kw) / stride;

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

  return qx;
}