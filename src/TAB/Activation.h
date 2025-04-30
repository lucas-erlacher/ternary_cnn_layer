#pragma once

#include "common.h"

// The simple Parameterized leaky ReLU function
// The tensor shape doesn't matter, because it apply PReLU on each value of the Conv Result
template <typename T>
void PReLU(T* x, int64_t N, int64_t C, int64_t H, int64_t W, float alpha, float* y) {

    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                        T current=x[((n * C + c) * H + h) * W + w];
                        if (current > 0)
                            y[((n * C + c) * H + h) * W + w] = current;
                        else
                            y[((n * C + c) * H + h) * W + w] = current * alpha;
                }
            }
        }
    }
}
