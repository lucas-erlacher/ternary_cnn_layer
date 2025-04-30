#include "../TAB/common.h"

static void inline pack_simd(float* tensor_in, int64_t* tensor_out, int ik, int channels, int priChannel, int packC, float q_threshold, int64_t* onebit);

void specialized_1x1_v1(float* tensor_x, float* tensor_w, float* res, int batch_size, int channels, int num_kernels, float q_threshold, float relu_alpha);
void specialized_1x1_v2(float* tensor_x, float* tensor_w, float* res, int batch_size, int channels, int num_kernels, float q_threshold, float relu_alpha);