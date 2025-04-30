## Optimization Overview
This folder contains all optimization steps that have been performed to make TNN faster. All functions that are cited in the submission paper can be found in the files.

### fused_1x1.cpp:
This file contains 2 versions for the 1x1 kernel. Both functions perform the whole TNN pipeline end-to-end. They use the knowledge gained from the general case and apply SIMD for packing. The data is stored in an intermediate buffer (since mostly, batch_size << KN). Afterwards, a for-loop iterates through the kernels, storing the channel values in a buffer. Then, GEMM is performed before PReLU is applied.

### fused_nhwc.cpp:
All versions that fuse quantization with img2row can be found in this file. The functions use increasing buffer sizes to copy larger chunks of memory from the data to the corresponding locations in the output.

### GEMM_opt.cpp:
This file contains all versions of ternary matrix multiplication (GEMM). The functions apply different strategies trying to increase the performance of popcounts per cycle, which is the identified bottleneck of the computation.

### yuliia_lucas_functions.cpp:
Contains implementations of fusing of ternarization and im2row in two variants (ternarize loop order and im2row loop order). for each variant of the fusing 3 versions with increasing optimzations were developed. descriptions of the functions (and their optimzations) can be found as comments over the function signatures. all functions from this file operate on tensors in NCHW format. Moreover, the file also contains helper functions for vectorized bit-packing.