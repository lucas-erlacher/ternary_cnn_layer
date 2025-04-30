#pragma once
#include "common.h"

template <typename T>
std::vector<T> Img2Row_NHWCB_to_N_OHOW_KHKWC(T* X, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t StrideH, int64_t StrideW) {

    const int64_t OH = (H - KH) / StrideH + 1;
    const int64_t OW = (W - KW) / StrideW + 1;
    const int64_t H1 = OH * OW;      // Fused Height
    const int64_t W1 = KH * KW * C;  // Fused Width
    std::vector<T> y = std::vector<T>(N * H1 * W1);

    for (int64_t n = 0; n < N; n++) {
        for (int64_t oh = 0; oh < OH; oh++) {
            for (int64_t ow = 0; ow < OW; ow++) {
                for (int64_t kh = 0; kh < KH; kh++) {
                    for (int64_t kw = 0; kw < KW; kw++) {
                        for (int64_t c = 0; c < C; c++)
                            // y[N, OH, OW, KH, KW, C] = X[N, H+kh, W+kw, C]
                            y[(n * H1 + oh * OW + ow) * W1 + kh * KW * C + kw * C + c] = X[((n * H + oh * StrideH + kh) * W + ow * StrideW + kw) * C + c];
                    }
                }
            }
        }
    }

    return y;
}