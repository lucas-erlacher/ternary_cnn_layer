#pragma once
std::vector<float> TAB_Conv(float* X, float Q_Threshold, int64_t* QWeights, int* BTN_CNT1, ConvType TYPE, int PaddingH, int PaddingW, int StrideH, int StrideW, int Batch_Size, int C, int H, int W, int KN, int KH, int KW, float ReLU_alpha);



