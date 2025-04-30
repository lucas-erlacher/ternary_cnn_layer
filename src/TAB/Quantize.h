#pragma once
std::vector<int64_t> Ternarize_NCHW_to_NHWCB(float* X, int64_t PaddingH, int64_t PaddingW, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W);
std::vector<int64_t> Binarize_NCHW_to_NHWC(const float* X, int PaddingH, int PaddingW, int N, int C, int H, int W);
std::vector<int64_t> Binarize_NCHW_to_NHWC(const float* X, int PaddingH, int PaddingW, float Q_Threshold, int N, int C, int H, int W);
std::vector<int> BTN_CNT_W2(int64_t* QW, int KN, int C, int KH, int KW);