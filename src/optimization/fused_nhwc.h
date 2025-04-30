#include "../TAB/common.h"

std::vector<int64_t> quantize_NHWC_to_NHWCB(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W);
std::vector<int64_t> quantize_NHWC_to_NHWCB_simd(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W);

std::vector<int64_t> fused_NHWC_to_NHWCB(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride);
std::vector<int64_t> fused_NHWC_to_NHWCB_simd(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride);
std::vector<int64_t> fused_NHWC_to_NHWCB_buffer(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride);
std::vector<int64_t> fused_NHWC_to_NHWCB_blocked(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t stride);
std::vector<int64_t> fused_NHWC_to_NHWCB_blocked3x3(float* X, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W);