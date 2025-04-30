#include "../TAB/common.h"

static inline void pack_SIMD(float* X, int64_t H, int64_t W, int64_t C, int64_t in, int64_t ic, int64_t ih, int64_t iw, float Q_Threshold, int64_t* onebit, int64_t& p1, int64_t& p2);
static inline void pack_SIMD_overhang(float* X, int64_t H, int64_t W, int64_t C, int64_t in, int64_t priChannel, int64_t ih, int64_t iw, float Q_Threshold, int64_t* onebit, int64_t& p1, int64_t& p2);

std::vector<int64_t> quantize_NCHW_to_NHWCB(float* X, int64_t padding, float Q_Threshold, int64_t N, int64_t C, int64_t H, int64_t W);

std::vector<int64_t> our_fusing_1(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t OH, int64_t OW);
std::vector<int64_t> our_fusing_2(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t OH, int64_t OW);
std::vector<int64_t> our_fusing_3(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW, int64_t OH, int64_t OW);

std::vector<int64_t> paper_fusing_1(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW);
std::vector<int64_t> paper_fusing_2(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW);
std::vector<int64_t> paper_fusing_3(float* X, float Q_Threshold, int64_t padding, int64_t stride, int64_t N, int64_t C, int64_t H, int64_t W, int64_t KH, int64_t KW);